"""Peptide-therapeutic deferred tools.

Practical reality (May 2026):

* **DRAMP** (http://dramp.cpu-bioinfor.org) and **THPdb** (https://webs.iiitd.edu.in/raghava/thpdb/)
  do **not** expose a public, documented JSON REST API — both are
  HTML-only with downloadable CSV/SDF dumps.  Wrapping them at runtime
  would require either HTML-scraping (fragile) or local ingestion of the
  bulk dumps (operational work outside the agent layer).

* **ChEMBL** *does* serve therapeutic peptides via the same
  ``/molecule.json`` endpoint when filtered by ``molecule_type``, and
  ``query_gtopdb`` already surfaces peptide ligands (IUPHAR has
  ~1,200 of them).  Together they cover the bulk of approved /
  late-stage peptide therapeutics.

The tools below therefore lean on ChEMBL.  When the agent needs DRAMP /
THPdb specifically (e.g. antimicrobial peptides), it should call
``query_pubmed`` for a literature route until the bulk dumps are
ingested into the local DB (Phase D, future work).
"""

from __future__ import annotations

import asyncio
import json
import logging

from langchain_core.tools import tool

from app.tools.base import query_rest_api
from app.tools.drug import _chembl_resolve_target  # reuse target-id resolver
from app.tools.preprocess import MAX_TOOL_TOKENS, guarded_tool

logger = logging.getLogger(__name__)


CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_chembl_peptides(target: str, max_results: int = 25) -> str:
    """Find peptide / oligopeptide drugs with reported activity against a target.

    Args:
        target: ChEMBL target id (``CHEMBL...``), UniProt accession, or
                gene symbol.  Resolution shares the small-molecule path.
        max_results: Cap on returned peptides.
    """
    target_id, target_name = await _chembl_resolve_target(target)
    if not target_id:
        return f"ChEMBL: could not resolve target {target!r}."

    # 1) Pull activities (any potency) and collect distinct molecule ids.
    try:
        act = await query_rest_api(
            f"{CHEMBL_BASE}/activity.json",
            params={
                "target_chembl_id": target_id,
                "limit": max(50, min(int(max_results) * 8, 200)),
            },
        )
    except Exception as exc:
        logger.exception("ChEMBL activity (peptide path) failed")
        return f"ChEMBL activity query failed: {exc}"

    candidate_ids: list[str] = []
    for a in act.get("activities", []) or []:
        mid = a.get("molecule_chembl_id")
        if mid and mid not in candidate_ids:
            candidate_ids.append(mid)

    # 2) Filter ChEMBL molecules for peptide / protein modality.
    # Limit candidates to avoid excessive API requests
    candidate_limit = max(50, max_results * 2)
    candidate_ids = candidate_ids[:candidate_limit]

    out: list[dict] = []
    sem = asyncio.Semaphore(5)

    async def _fetch_peptide(mid: str) -> dict | None:
        async with sem:
            try:
                mol = await query_rest_api(f"{CHEMBL_BASE}/molecule/{mid}.json")
            except Exception as exc:
                logger.debug("Molecule lookup failed for %s: %s", mid, exc)
                return None
            if not isinstance(mol, dict):
                return None
            m_type = (mol.get("molecule_type") or "").lower()
            if not any(k in m_type for k in ("peptide", "oligopeptide", "protein")):
                return None
            struct = mol.get("molecule_structures") or {}
            seq = struct.get("sequence") or struct.get("helm_notation") or None
            return {
                "molecule_chembl_id": mid,
                "pref_name": mol.get("pref_name"),
                "molecule_type": mol.get("molecule_type"),
                "max_phase": mol.get("max_phase"),
                "peptide_sequence": seq,
                "canonical_smiles": struct.get("canonical_smiles"),
                "url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{mid}/",
            }

    fetched_mols = await asyncio.gather(*[_fetch_peptide(mid) for mid in candidate_ids])
    for mol_data in fetched_mols:
        if mol_data is not None:
            out.append(mol_data)
        if len(out) >= max_results:
            break

    return json.dumps(
        {
            "target_chembl_id": target_id,
            "target_name": target_name,
            "peptides": out,
            "count": len(out),
            "note": "DRAMP/THPdb not wrapped (no public REST). Use literature search for AMPs.",
        },
        ensure_ascii=False,
        indent=2,
    )
