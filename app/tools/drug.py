"""Drug-discovery deferred tools: ChEMBL target activities, PubChem, IUPHAR/GtoPdb."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
from langchain_core.tools import tool

from app.tools.base import query_rest_api
from app.tools.preprocess import MAX_TOOL_TOKENS, guarded_tool

logger = logging.getLogger(__name__)


# --- ChEMBL target → activities --------------------------------------

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


async def _chembl_resolve_target(
    query: str,
) -> tuple[str | None, str | None]:
    """Resolve a free-form target string (UniProt acc, gene symbol, or
    ChEMBL target id) to a ``target_chembl_id`` + display name."""
    q = query.strip()
    if q.upper().startswith("CHEMBL"):
        return q.upper(), None
    # Try UniProt accession xref first.
    if q.isalnum() and 6 <= len(q) <= 10 and q.isupper():
        try:
            data = await query_rest_api(
                f"{CHEMBL_BASE}/target.json",
                params={
                    "target_components__accession": q,
                    "limit": 5,
                },
            )
            for t in data.get("targets", []) or []:
                if t.get("target_chembl_id"):
                    return t["target_chembl_id"], t.get("pref_name")
        except Exception as exc:
            logger.warning("ChEMBL target by accession failed: %s", exc)

    # Fall back to free-text search.
    try:
        data = await query_rest_api(
            f"{CHEMBL_BASE}/target/search.json",
            params={"q": q, "limit": 5},
        )
        for t in data.get("targets", []) or []:
            if t.get("target_chembl_id"):
                return t["target_chembl_id"], t.get("pref_name")
    except Exception as exc:
        logger.warning("ChEMBL target search failed: %s", exc)
    return None, None


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_chembl_target_activities(
    target: str,
    activity_type: str = "IC50",
    pchembl_min: float = 6.0,
    max_results: int = 25,
) -> str:
    """List bioactive compounds against a target with potency above a threshold.

    Args:
        target: ChEMBL target id (``CHEMBL203``), UniProt accession
                (``P00533``), or gene symbol (``EGFR``).
        activity_type: ``IC50`` / ``Ki`` / ``Kd`` / ``EC50``.
        pchembl_min: Minimum pChEMBL value (6 = 1 µM, 7 = 100 nM, 8 = 10 nM).
        max_results: Cap on returned compounds.
    """
    target_id, target_name = await _chembl_resolve_target(target)
    if not target_id:
        return f"ChEMBL: could not resolve target {target!r}."

    params = {
        "target_chembl_id": target_id,
        "standard_type": activity_type.upper(),
        "pchembl_value__gte": pchembl_min,
        "limit": max(1, min(int(max_results), 100)),
    }
    try:
        data = await query_rest_api(
            f"{CHEMBL_BASE}/activity.json",
            params=params,
        )
    except Exception as exc:
        logger.exception("ChEMBL activity query failed")
        return f"ChEMBL activity query failed: {exc}"

    activities = data.get("activities", []) or []
    rows: list[dict[str, Any]] = []
    seen_mols: set[str] = set()
    for a in activities:
        mid = a.get("molecule_chembl_id")
        if not mid or mid in seen_mols:
            continue
        seen_mols.add(mid)
        rows.append(
            {
                "molecule_chembl_id": mid,
                "pref_name": a.get("molecule_pref_name"),
                "canonical_smiles": a.get("canonical_smiles"),
                "type": a.get("standard_type"),
                "value": a.get("standard_value"),
                "units": a.get("standard_units"),
                "pchembl": a.get("pchembl_value"),
                "assay_description": (a.get("assay_description") or "")[:200],
            }
        )
    return json.dumps(
        {
            "target_chembl_id": target_id,
            "target_name": target_name,
            "activity_type": activity_type.upper(),
            "pchembl_min": pchembl_min,
            "compounds": rows,
            "count": len(rows),
        },
        ensure_ascii=False,
        indent=2,
    )


# --- PubChem ----------------------------------------------------------

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_pubchem(query: str, namespace: str = "name") -> str:
    """Look up a compound in PubChem and return its core identifiers.

    Args:
        query: The compound name, CID, SMILES, or InChIKey.
        namespace: One of ``name`` / ``cid`` / ``smiles`` / ``inchikey``.
    """
    q = query.strip()
    ns = namespace.lower()
    props = "MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChIKey,IUPACName"
    url = f"{PUBCHEM_BASE}/compound/{ns}/{q}/property/{props}/JSON"
    try:
        data = await query_rest_api(url)
    except Exception as exc:
        logger.exception("PubChem query failed")
        return f"PubChem query failed for {query!r}: {exc}"
    props_list = (data.get("PropertyTable") or {}).get("Properties") or []
    if not props_list:
        return f"No PubChem hit for {query!r}."
    out = []
    for p in props_list[:5]:
        out.append(
            {
                "cid": p.get("CID"),
                "iupac_name": p.get("IUPACName"),
                "formula": p.get("MolecularFormula"),
                "mw": p.get("MolecularWeight"),
                "canonical_smiles": p.get("CanonicalSMILES"),
                "isomeric_smiles": p.get("IsomericSMILES"),
                "inchikey": p.get("InChIKey"),
                "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{p.get('CID')}"
                if p.get("CID")
                else None,
            }
        )
    return json.dumps({"query": query, "compounds": out}, ensure_ascii=False, indent=2)


# --- IUPHAR / Guide to Pharmacology ----------------------------------

GTOPDB_BASE = "https://www.guidetopharmacology.org/services"


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_gtopdb(target_name: str, max_results: int = 25) -> str:
    """Find IUPHAR/GtoPdb targets and their bound ligands by name.

    Returns target metadata + linked ligand list (small molecules AND
    peptides — the IUPHAR DB has both).

    Args:
        target_name: Free-text target name (``"EGFR"``, ``"GLP-1 receptor"``).
        max_results: Cap on returned ligands.
    """
    name = target_name.strip()
    try:
        targets = await query_rest_api(
            f"{GTOPDB_BASE}/targets",
            params={"name": name},
        )
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return f"GtoPdb: target {name!r} not found in IUPHAR database."
        logger.exception("GtoPdb target query failed")
        return f"GtoPdb query failed for {name!r}: {exc}"
    except Exception as exc:
        logger.exception("GtoPdb target query failed")
        return f"GtoPdb query failed for {name!r}: {exc}"
    if not isinstance(targets, list) or not targets:
        return f"GtoPdb: no targets found for {name!r}."

    target = targets[0]
    target_id = target.get("targetId")

    # Fetch ligand interactions for the target.
    try:
        interactions = await query_rest_api(
            f"{GTOPDB_BASE}/targets/{target_id}/interactions"
        )
    except Exception as exc:
        logger.warning("GtoPdb interactions failed: %s", exc)
        interactions = []

    ligand_ids: list[int] = []
    for it in interactions or []:
        lid = it.get("ligandId")
        if lid and lid not in ligand_ids:
            ligand_ids.append(lid)
        if len(ligand_ids) >= max_results:
            break

    sem = asyncio.Semaphore(5)

    async def _fetch_ligand(lid: int) -> dict[str, Any] | None:
        async with sem:
            try:
                lig = await query_rest_api(f"{GTOPDB_BASE}/ligands/{lid}")
            except Exception as exc:
                logger.warning("GtoPdb ligand fetch failed for %s: %s", lid, exc)
                return None
        if not isinstance(lig, dict):
            return None
        l_type = (lig.get("type") or "").lower()
        if "peptide" in l_type or "antibody" in l_type or "protein" in l_type:
            modality = "peptide" if "peptide" in l_type else "other"
        elif "synthetic organic" in l_type or "natural product" in l_type:
            modality = "small_molecule"
        else:
            modality = "small_molecule"
        return {
            "gtopdb_id": lid,
            "name": lig.get("name"),
            "type": lig.get("type"),
            "modality": modality,
            "smiles": lig.get("smiles"),
            "approved": lig.get("approved"),
            "url": f"https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId={lid}",
        }

    fetched = await asyncio.gather(*[_fetch_ligand(lid) for lid in ligand_ids])
    ligands_out: list[dict[str, Any]] = [r for r in fetched if r is not None]

    return json.dumps(
        {
            "target_id": target_id,
            "target_name": target.get("name"),
            "target_family": target.get("familyIds"),
            "ligands": ligands_out,
            "count": len(ligands_out),
        },
        ensure_ascii=False,
        indent=2,
    )
