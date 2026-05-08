"""Target Discovery service.

Bridges the LangGraph sub-graph and the relational store:

* ``discover_target``  — runs the agent, persists the report, returns it.
* ``persist_report``   — pure DB upsert (idempotent on ``(target, version)``).
* ``get_target``       — fetch by id with ORM relationships eager-loaded.
* ``get_latest_report``— most recent ``TargetReport`` for a target.
* ``list_targets``     — paginated user view.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent.llm_provider import reset_failed_models
from app.agent.target_discovery_graph import run_target_discovery
from app.models.pathway_drug import (
    DiseaseAssociation,
    Drug,
    Paper,
    Pathway,
    TargetDrugActivity,
)
from app.models.target import ProteinRecord, Target
from app.models.target_report import TargetReport

logger = logging.getLogger(__name__)


# --- internal helpers ------------------------------------------------


async def _upsert_target(
    db: AsyncSession,
    *,
    name: str,
    gene_symbol: str | None,
    organism: str,
    description: str | None,
    uniprot_ids: list[str],
) -> Target:
    """Find by (name, organism); create if missing. Refreshes uniprot_ids."""
    result = await db.execute(
        select(Target).where(Target.name == name, Target.organism == organism)
    )
    target = result.scalar_one_or_none()
    if target is None:
        target = Target(
            name=name,
            gene_symbol=gene_symbol,
            organism=organism,
            description=description,
            uniprot_ids=list(uniprot_ids or []),
        )
        db.add(target)
        await db.flush()
        return target
    # Merge new info without dropping prior knowledge.
    if gene_symbol and not target.gene_symbol:
        target.gene_symbol = gene_symbol
    if description and not target.description:
        target.description = description
    merged = list({*target.uniprot_ids, *(uniprot_ids or [])})
    if merged != target.uniprot_ids:
        target.uniprot_ids = merged
    return target


async def _upsert_protein(db: AsyncSession, target: Target, p: dict[str, Any]) -> None:
    uid = (p.get("accession") or p.get("uniprot_id") or "").upper()
    if not uid:
        return
    result = await db.execute(
        select(ProteinRecord).where(
            ProteinRecord.target_id == target.id,
            ProteinRecord.uniprot_id == uid,
        )
    )
    rec = result.scalar_one_or_none()
    if rec is None:
        rec = ProteinRecord(target_id=target.id, uniprot_id=uid)
        db.add(rec)
    rec.name = p.get("name") or rec.name
    rec.gene = p.get("gene") or rec.gene
    rec.sequence_length = p.get("sequence_length") or rec.sequence_length
    if p.get("sequence"):
        rec.sequence = p["sequence"]
    if p.get("pdb_ids"):
        rec.pdb_ids = list({*(rec.pdb_ids or []), *p["pdb_ids"]})
    if p.get("alphafold_id"):
        rec.alphafold_id = p["alphafold_id"]
    if p.get("interpro_domains"):
        rec.interpro_domains = list({*(rec.interpro_domains or []), *p["interpro_domains"]})


async def _upsert_pathway(db: AsyncSession, target: Target, p: dict[str, Any]) -> None:
    src = p.get("source")
    eid = p.get("external_id")
    if not src or not eid:
        return
    result = await db.execute(
        select(Pathway).where(Pathway.source == src, Pathway.external_id == eid)
    )
    pw = result.scalar_one_or_none()
    if pw is None:
        pw = Pathway(
            source=src,
            external_id=eid,
            name=p.get("name") or eid,
            description=p.get("description"),
            url=p.get("url"),
        )
        db.add(pw)
        await db.flush()
    if target not in pw.targets:
        pw.targets.append(target)


async def _upsert_paper(db: AsyncSession, target: Target, p: dict[str, Any]) -> None:
    pmid = p.get("pmid")
    doi = p.get("doi")
    if not (pmid or doi or p.get("title")):
        return
    paper: Paper | None = None
    if pmid:
        result = await db.execute(select(Paper).where(Paper.pmid == pmid))
        paper = result.scalar_one_or_none()
    if paper is None and doi:
        result = await db.execute(select(Paper).where(Paper.doi == doi))
        paper = result.scalar_one_or_none()
    if paper is None:
        paper = Paper(
            pmid=pmid,
            doi=doi,
            title=p.get("title") or "Untitled",
            year=p.get("year"),
            journal=p.get("journal"),
            url=p.get("url"),
            extra={k: v for k, v in p.items() if k not in {"pmid", "doi", "title", "year", "journal", "url"}},
        )
        db.add(paper)
        await db.flush()
    if target not in paper.targets:
        paper.targets.append(target)


async def _upsert_disease(db: AsyncSession, target: Target, d: dict[str, Any]) -> None:
    if not d.get("disease_id") or not d.get("disease_name"):
        return
    result = await db.execute(
        select(DiseaseAssociation).where(
            DiseaseAssociation.target_id == target.id,
            DiseaseAssociation.disease_id == d["disease_id"],
            DiseaseAssociation.source == (d.get("source") or "OpenTargets"),
        )
    )
    if result.scalar_one_or_none() is not None:
        return
    db.add(
        DiseaseAssociation(
            target_id=target.id,
            source=d.get("source") or "OpenTargets",
            disease_id=d["disease_id"],
            disease_name=d["disease_name"],
            score=d.get("score"),
            evidence_summary=d.get("evidence_summary"),
            url=d.get("url"),
        )
    )


async def _upsert_drug(
    db: AsyncSession,
    target: Target,
    raw: dict[str, Any],
    *,
    modality: str,
) -> None:
    chembl_id = raw.get("chembl_id") or raw.get("molecule_chembl_id")
    drug: Drug | None = None
    if chembl_id:
        result = await db.execute(select(Drug).where(Drug.chembl_id == chembl_id))
        drug = result.scalar_one_or_none()
    if drug is None:
        drug = Drug(
            chembl_id=chembl_id,
            name=raw.get("name") or raw.get("pref_name"),
            modality=modality,
            smiles=raw.get("smiles") or raw.get("canonical_smiles"),
            inchikey=raw.get("inchikey") or raw.get("standard_inchi_key"),
            peptide_sequence=raw.get("sequence") or raw.get("peptide_sequence"),
            max_phase=raw.get("max_phase"),
            mechanism_of_action=raw.get("mechanism_of_action"),
        )
        db.add(drug)
        await db.flush()

    activity = raw.get("activity") or {}
    if not activity and raw.get("pchembl"):
        activity = {
            "type": raw.get("type"),
            "value_nm": raw.get("value"),
            "pchembl": raw.get("pchembl"),
            "assay": raw.get("assay_description"),
        }
    if activity:
        atype = activity.get("type")
        result = await db.execute(
            select(TargetDrugActivity).where(
                TargetDrugActivity.target_id == target.id,
                TargetDrugActivity.drug_id == drug.id,
                TargetDrugActivity.activity_type == atype,
            )
        )
        if result.scalar_one_or_none() is None:
            db.add(
                TargetDrugActivity(
                    target_id=target.id,
                    drug_id=drug.id,
                    activity_type=atype,
                    value_nm=activity.get("value_nm"),
                    pchembl=activity.get("pchembl"),
                    assay_description=activity.get("assay") or activity.get("assay_description"),
                    source="ChEMBL" if chembl_id else None,
                )
            )


# --- public API ------------------------------------------------------


async def persist_report(
    db: AsyncSession,
    *,
    report: dict[str, Any],
    user_id: uuid.UUID | None = None,
    session_id: uuid.UUID | None = None,
) -> TargetReport:
    """Idempotently store a TargetReport dict + all extracted entities."""
    t_dict = report.get("target") or {}
    name = t_dict.get("name") or report.get("target_query") or "Unknown target"
    target = await _upsert_target(
        db,
        name=name,
        gene_symbol=t_dict.get("gene_symbol"),
        organism=t_dict.get("organism") or "Homo sapiens",
        description=t_dict.get("description"),
        uniprot_ids=t_dict.get("uniprot_ids") or [],
    )

    for p in report.get("proteins") or []:
        await _upsert_protein(db, target, p)
    for paper in report.get("papers") or []:
        await _upsert_paper(db, target, paper)
    for d in report.get("disease_associations") or []:
        await _upsert_disease(db, target, d)
    for pw in report.get("pathways") or []:
        await _upsert_pathway(db, target, pw)
    for sm in report.get("small_molecule_drugs") or []:
        await _upsert_drug(db, target, sm, modality="small_molecule")
    for pep in report.get("peptide_drugs") or []:
        await _upsert_drug(db, target, pep, modality="peptide")

    # Latest version + 1
    result = await db.execute(
        select(TargetReport.version)
        .where(TargetReport.target_id == target.id)
        .order_by(TargetReport.version.desc())
        .limit(1)
    )
    last_version = result.scalar_one_or_none() or 0
    snapshot = TargetReport(
        target_id=target.id,
        session_id=session_id,
        user_id=user_id,
        version=last_version + 1,
        content=report,
        notes=list(report.get("notes") or []),
    )
    db.add(snapshot)
    await db.commit()
    await db.refresh(snapshot)
    return snapshot


async def discover_target(
    db: AsyncSession,
    *,
    provider: Any,
    target_query: str,
    user_id: uuid.UUID | None = None,
    session_id: uuid.UUID | None = None,
) -> TargetReport:
    """Run the discovery sub-graph end-to-end and persist the result."""
    reset_failed_models()
    report = await run_target_discovery(provider, target_query)
    return await persist_report(
        db, report=report, user_id=user_id, session_id=session_id
    )


async def list_targets(
    db: AsyncSession, *, limit: int = 50, offset: int = 0
) -> list[Target]:
    result = await db.execute(
        select(Target)
        .order_by(Target.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_target(db: AsyncSession, target_id: uuid.UUID) -> Target | None:
    result = await db.execute(
        select(Target)
        .options(
            selectinload(Target.proteins),
            selectinload(Target.pathways),
            selectinload(Target.papers),
        )
        .where(Target.id == target_id)
    )
    return result.scalar_one_or_none()


async def get_latest_report(
    db: AsyncSession, target_id: uuid.UUID
) -> TargetReport | None:
    result = await db.execute(
        select(TargetReport)
        .where(TargetReport.target_id == target_id)
        .order_by(TargetReport.version.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()
