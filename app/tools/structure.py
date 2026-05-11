"""Structure-related deferred tools: PDB / AlphaFold / InterPro.

All thin async REST wrappers — output is a compact markdown table or a
structured Pydantic model serialised to JSON. The heavy raw payload is
NEVER returned to the LLM (design doc §7.2.1, hard pruning).
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from app.tools.base import query_rest_api
from app.tools.preprocess import MAX_TOOL_TOKENS, guarded_tool

logger = logging.getLogger(__name__)


# --- RCSB PDB ---------------------------------------------------------

PDB_DATA_BASE = "https://data.rcsb.org/rest/v1/core"
PDB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_pdb(pdb_id: str) -> str:
    """Look up a single PDB entry (4-character ID) and return its key
    metadata: title, resolution, experimental method, and the polymer
    entity composition (chains + UniProt cross-refs + length).

    Args:
        pdb_id: A 4-character RCSB PDB identifier (e.g. ``1M17`` for EGFR).
    """
    pid = pdb_id.strip().upper()
    if len(pid) != 4 or not pid.isalnum():
        return f"Invalid PDB id: {pdb_id!r}; expected 4 alphanumeric chars."

    try:
        entry = await query_rest_api(f"{PDB_DATA_BASE}/entry/{pid}")
    except Exception as exc:
        logger.exception("PDB entry fetch failed")
        return f"PDB query failed for {pid}: {exc}"

    title = (entry.get("struct") or {}).get("title")
    resolution = None
    rs = entry.get("rcsb_entry_info") or {}
    if "resolution_combined" in rs and rs["resolution_combined"]:
        resolution = rs["resolution_combined"][0]
    method = ", ".join(rs.get("experimental_method", []) or [])

    polymer_ids = (entry.get("rcsb_entry_container_identifiers") or {}).get(
        "polymer_entity_ids", []
    ) or []

    entities: list[dict] = []
    for eid in polymer_ids[:8]:
        try:
            ent = await query_rest_api(f"{PDB_DATA_BASE}/polymer_entity/{pid}/{eid}")
        except Exception as exc:
            logger.warning("polymer_entity fetch failed for %s/%s: %s", pid, eid, exc)
            continue
        chains = (ent.get("rcsb_polymer_entity_container_identifiers") or {}).get(
            "asym_ids", []
        ) or []
        seq_len = (ent.get("entity_poly") or {}).get("rcsb_sample_sequence_length")
        # UniProt cross-refs
        uniprot_ids: list[str] = []
        for ref in ent.get("rcsb_polymer_entity_container_identifiers", {}).get(
            "reference_sequence_identifiers", []
        ) or []:
            if ref.get("database_name") == "UniProt" and ref.get("database_accession"):
                uniprot_ids.append(ref["database_accession"])
        entities.append(
            {
                "entity_id": eid,
                "name": (ent.get("rcsb_polymer_entity") or {}).get("pdbx_description"),
                "chains": chains,
                "length": seq_len,
                "uniprot": uniprot_ids,
            }
        )

    payload = {
        "pdb_id": pid,
        "title": title,
        "resolution_A": resolution,
        "experimental_method": method,
        "entities": entities,
        "url": f"https://www.rcsb.org/structure/{pid}",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_pdb_identifiers(uniprot_id: str, max_results: int = 25) -> str:
    """Search RCSB for PDB entries containing a given UniProt accession.

    Returns a JSON list of matching ``pdb_id`` strings — useful when the
    UniProt entry's xref list is incomplete or you want a fresh view.

    Args:
        uniprot_id: UniProt accession (e.g. ``P00533``).
        max_results: Cap on returned ids.
    """
    uid = uniprot_id.strip().upper()
    body = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": (
                    "rcsb_polymer_entity_container_identifiers."
                    "reference_sequence_identifiers.database_accession"
                ),
                "operator": "exact_match",
                "value": uid,
            },
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max(1, min(int(max_results), 100))}
        },
    }
    try:
        data = await query_rest_api(
            PDB_SEARCH_URL,
            method="POST",
            json_body=body,
            headers={"Content-Type": "application/json"},
            use_cache=True,
        )
    except Exception as exc:
        logger.exception("PDB search failed")
        return f"PDB search failed: {exc}"
    ids = [r.get("identifier") for r in data.get("result_set", []) if r.get("identifier")]
    return json.dumps({"uniprot": uid, "pdb_ids": ids, "total": len(ids)})


# --- AlphaFold --------------------------------------------------------

ALPHAFOLD_BASE = "https://alphafold.ebi.ac.uk/api"


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_alphafold(uniprot_id: str) -> str:
    """Fetch the AlphaFold-DB predicted structure metadata for a UniProt
    accession. Returns model URL, confidence, and PAE plot URL.

    Args:
        uniprot_id: UniProt accession (e.g. ``P00533``).
    """
    uid = uniprot_id.strip().upper()
    try:
        data = await query_rest_api(f"{ALPHAFOLD_BASE}/prediction/{uid}")
    except Exception as exc:
        logger.exception("AlphaFold query failed")
        return f"AlphaFold query failed for {uid}: {exc}"
    if not isinstance(data, list) or not data:
        return f"No AlphaFold prediction available for {uid}."
    item = data[0]
    payload = {
        "uniprot_id": uid,
        "model_url": item.get("pdbUrl"),
        "cif_url": item.get("cifUrl"),
        "pae_image_url": item.get("paeImageUrl"),
        "pae_doc_url": item.get("paeDocUrl"),
        "model_created_date": item.get("modelCreatedDate"),
        "uniprot_description": item.get("uniprotDescription"),
        "organism": item.get("organismScientificName"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# --- InterPro ---------------------------------------------------------

INTERPRO_BASE = "https://www.ebi.ac.uk/interpro/api"


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_interpro(uniprot_id: str) -> str:
    """List InterPro domain annotations for a UniProt protein.

    Args:
        uniprot_id: UniProt accession (e.g. ``P00533``).
    """
    uid = uniprot_id.strip().upper()
    url = f"{INTERPRO_BASE}/entry/interpro/protein/uniprot/{uid}/?page_size=50"
    try:
        data = await query_rest_api(url)
    except Exception as exc:
        logger.exception("InterPro query failed")
        return f"InterPro query failed for {uid}: {exc}"
    results = data.get("results") or []
    domains = []
    for r in results:
        meta = r.get("metadata") or {}
        domains.append(
            {
                "interpro_id": meta.get("accession"),
                "name": meta.get("name"),
                "type": meta.get("type"),
            }
        )
    return json.dumps(
        {"uniprot": uid, "domains": domains, "count": len(domains)},
        ensure_ascii=False,
        indent=2,
    )
