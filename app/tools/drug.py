"""Drug-discovery deferred tools: ChEMBL target activities, PubChem, IUPHAR/GtoPdb."""

from __future__ import annotations

import asyncio
import json
import logging
from contextvars import ContextVar
from typing import Any

import httpx
from langchain_core.tools import tool

from app.tools.base import query_rest_api
from app.tools.preprocess import MAX_TOOL_TOKENS, guarded_tool

logger = logging.getLogger(__name__)


# --- ChEMBL target → activities --------------------------------------

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Per-task cache of already-queried (target, activity_types_set) pairs.
# Propagates across asyncio tasks within the same node, so each drugs-node
# run gets its own isolated cache. Prevents the LLM from burning the
# 180s budget by calling the tool 3-4 times sequentially with one
# activity_type each (a pattern observed in production despite prompt rules).
_chembl_call_cache: ContextVar[dict[str, str] | None] = ContextVar(
    "chembl_call_cache", default=None
)
# Default fan-out used whenever the LLM asks for a single activity_type:
# we expand it on the LLM's behalf so the next call (if any) hits cache.
_DEFAULT_FANOUT_TYPES = ["IC50", "KI", "KD", "EC50"]


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
        activity_type: One assay type (``IC50``/``Ki``/``Kd``/``EC50``) **or**
                a comma-separated list to fetch multiple types in parallel
                in a SINGLE tool call (e.g. ``"IC50,Ki,Kd,EC50"``). Strongly
                preferred over calling this tool 4 times sequentially —
                serial calls easily blow the per-node 180s budget.
        pchembl_min: Minimum pChEMBL value (6 = 1 µM, 7 = 100 nM, 8 = 10 nM).
        max_results: Cap on returned compounds (applied across all types
                combined; deduplicated by ``molecule_chembl_id``).
    """
    target_id, target_name = await _chembl_resolve_target(target)
    if not target_id:
        return f"ChEMBL: could not resolve target {target!r}."

    types = [t.strip().upper() for t in activity_type.split(",") if t.strip()]
    if not types:
        types = ["IC50"]
    # AUTO-EXPAND: if the LLM asked for a single type, fan out to the full
    # default set so subsequent calls hit cache. This protects the per-node
    # time budget against LLMs that ignore the prompt's batching rule.
    if len(types) == 1:
        types = list(_DEFAULT_FANOUT_TYPES)
    # Per-type result cap; final result is deduped+capped to ``max_results``.
    per_type_limit = max(1, min(int(max_results), 100))

    # Per-task cache: serve repeat calls (same target, any subset of
    # already-fetched types) from memory. Works because ContextVar is
    # propagated across tasks spawned within the same node.
    cache = _chembl_call_cache.get()
    if cache is None:
        cache = {}
        _chembl_call_cache.set(cache)
    cache_key = target_id  # one entry per target id per task
    if cache_key in cache:
        return (
            "NOTE: query_chembl_target_activities was already called for this "
            f"target (`{target_id}`) earlier in this node; reusing prior result. "
            "Do not call this tool again for the same target.\n\n" + cache[cache_key]
        )

    async def _fetch_one(act_type: str) -> tuple[str, list[dict[str, Any]] | str]:
        params = {
            "target_chembl_id": target_id,
            "standard_type": act_type,
            "pchembl_value__gte": pchembl_min,
            "limit": per_type_limit,
        }
        try:
            data = await query_rest_api(
                f"{CHEMBL_BASE}/activity.json",
                params=params,
            )
        except Exception as exc:
            logger.warning("ChEMBL activity query failed for %s: %s", act_type, exc)
            return act_type, f"error: {exc}"
        return act_type, data.get("activities", []) or []

    # Parallel fan-out across activity types.
    fetched = await asyncio.gather(*[_fetch_one(t) for t in types])

    rows: list[dict[str, Any]] = []
    seen_mols: set[str] = set()
    errors: dict[str, str] = {}
    for act_type, payload in fetched:
        if isinstance(payload, str):
            errors[act_type] = payload
            continue
        for a in payload:
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
            if len(rows) >= max_results:
                break
        if len(rows) >= max_results:
            break

    out = {
        "target_chembl_id": target_id,
        "target_name": target_name,
        "activity_types": types,
        "pchembl_min": pchembl_min,
        "compounds": rows,
        "count": len(rows),
    }
    if errors:
        out["errors_by_type"] = errors
    serialized = json.dumps(out, ensure_ascii=False, indent=2)
    # Cache for the rest of this node-task (mutate the dict in place — the
    # ContextVar holds a reference so all spawned tasks see the update).
    cache[cache_key] = serialized
    _chembl_call_cache.set(cache)
    return serialized


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
        # IUPHAR/GtoPdb intermittently returns 503; bump retries so a flaky
        # window doesn't blank out the drugs node.
        targets = await query_rest_api(
            f"{GTOPDB_BASE}/targets",
            params={"name": name},
            max_retries=5,
        )
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return f"GtoPdb: target {name!r} not found in IUPHAR database."
        code = exc.response.status_code if exc.response is not None else 0
        logger.exception("GtoPdb target query failed")
        # Structured error so the LLM sees "retryable" and doesn't conclude
        # "no GtoPdb data exists for this target".
        return json.dumps(
            {"source": "GtoPdb", "target_name": name, "error": f"HTTP {code}", "retryable": True},
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.exception("GtoPdb target query failed")
        return json.dumps(
            {"source": "GtoPdb", "target_name": name, "error": str(exc), "retryable": True},
            ensure_ascii=False,
        )
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
