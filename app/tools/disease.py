"""Disease / function tools: OpenTargets / Monarch / QuickGO."""

from __future__ import annotations

import json
import logging

import httpx

from langchain_core.tools import tool

from app.tools.base import query_rest_api
from app.tools.preprocess import MAX_TOOL_TOKENS, guarded_tool

logger = logging.getLogger(__name__)


# --- OpenTargets (GraphQL) -------------------------------------------

OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# Two narrow, parameterised queries — covers 95% of "target → diseases"
# and "disease → targets" use-cases without exposing the raw schema.
_TARGET_DISEASES_QUERY = """
query TargetDiseases($ensemblId: String!, $size: Int!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    biotype
    associatedDiseases(page: {index: 0, size: $size}) {
      count
      rows {
        score
        disease {
          id
          name
          therapeuticAreas { name }
        }
      }
    }
  }
}
"""

_TARGET_BY_SYMBOL_QUERY = """
query TargetSearch($q: String!) {
  search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 5}) {
    hits {
      id
      name
      entity
      object {
        ... on Target {
          id
          approvedSymbol
          approvedName
        }
      }
    }
  }
}
"""


async def _opentargets_post(query: str, variables: dict) -> dict:
    return await query_rest_api(
        OPENTARGETS_URL,
        method="POST",
        json_body={"query": query, "variables": variables},
        headers={"Content-Type": "application/json"},
        use_cache=True,
    )


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_opentarget(target: str, max_results: int = 25) -> str:
    """Find diseases associated with a target via OpenTargets Platform.

    Args:
        target: Either an Ensembl gene id (``ENSG...``) or a gene symbol
                (``EGFR``). Symbols are resolved via the search endpoint.
        max_results: Cap on returned disease associations.
    """
    t = target.strip()
    ensembl_id: str | None = None
    if t.upper().startswith("ENSG"):
        ensembl_id = t.upper()
    else:
        try:
            data = await _opentargets_post(_TARGET_BY_SYMBOL_QUERY, {"q": t})
        except Exception as exc:
            return f"OpenTargets symbol resolve failed: {exc}"
        for hit in (data.get("data", {}).get("search") or {}).get("hits", []) or []:
            obj = hit.get("object") or {}
            if (obj.get("approvedSymbol") or "").upper() == t.upper():
                ensembl_id = obj.get("id")
                break
        if not ensembl_id:
            hits = (data.get("data", {}).get("search") or {}).get("hits", []) or []
            if hits:
                ensembl_id = (hits[0].get("object") or {}).get("id")
        if not ensembl_id:
            return f"OpenTargets: no target found for {target!r}."

    try:
        payload = await _opentargets_post(
            _TARGET_DISEASES_QUERY,
            {"ensemblId": ensembl_id, "size": max(1, min(int(max_results), 50))},
        )
    except Exception as exc:
        logger.exception("OpenTargets associations failed")
        return f"OpenTargets query failed: {exc}"

    target_block = (payload.get("data") or {}).get("target") or {}
    rows = (target_block.get("associatedDiseases") or {}).get("rows", []) or []

    out = {
        "ensembl_id": ensembl_id,
        "approved_symbol": target_block.get("approvedSymbol"),
        "approved_name": target_block.get("approvedName"),
        "biotype": target_block.get("biotype"),
        "associations": [
            {
                "disease_id": (r.get("disease") or {}).get("id"),
                "disease_name": (r.get("disease") or {}).get("name"),
                "score": r.get("score"),
                "therapeutic_areas": [
                    ta.get("name") for ta in (r.get("disease") or {}).get("therapeuticAreas", [])
                ],
            }
            for r in rows
        ],
        "count": len(rows),
        "url": f"https://platform.opentargets.org/target/{ensembl_id}",
    }
    return json.dumps(out, ensure_ascii=False, indent=2)


# --- Monarch Initiative ----------------------------------------------

MONARCH_BASE = "https://api.monarchinitiative.org/v3/api"


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_monarch(query: str, category: str = "biolink:Disease", max_results: int = 10) -> str:
    """Search Monarch Initiative for entities (diseases / phenotypes / genes).

    Args:
        query: Free-text query (e.g. ``"EGFR lung cancer"``, ``"marfan"``).
        category: Biolink category filter — ``biolink:Disease``,
                  ``biolink:PhenotypicFeature``, ``biolink:Gene``...
        max_results: Cap on returned hits.
    """
    params = {
        "q": query,
        "category": category,
        "limit": max(1, min(int(max_results), 50)),
    }
    try:
        data = await query_rest_api(f"{MONARCH_BASE}/search", params=params)
    except Exception as exc:
        logger.exception("Monarch search failed")
        return f"Monarch query failed: {exc}"
    items = data.get("items") or data.get("results") or []
    rows = []
    for it in items:
        rows.append(
            {
                "id": it.get("id"),
                "name": it.get("name"),
                "category": it.get("category"),
                "description": (it.get("description") or "")[:300],
            }
        )
    return json.dumps(
        {"query": query, "category": category, "results": rows, "count": len(rows)},
        ensure_ascii=False,
        indent=2,
    )


# --- QuickGO (Gene Ontology) -----------------------------------------

QUICKGO_BASE = "https://www.ebi.ac.uk/QuickGO/services"


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_quickgo(uniprot_id: str, aspect: str = "F", max_results: int = 30) -> str:
    """Retrieve GO term annotations for a UniProt protein.

    Args:
        uniprot_id: UniProt accession (e.g. ``P00533``).
        aspect: GO aspect — ``F`` molecular_function, ``P`` biological_process,
                ``C`` cellular_component. Pass ``"FPC"`` for all.
        max_results: Cap on returned annotations.
    """
    uid = uniprot_id.strip().upper()
    aspect_map = {"F": "molecular_function", "P": "biological_process", "C": "cellular_component"}
    aspects = ",".join(aspect_map.get(a, "") for a in aspect.upper() if a in aspect_map)
    params: dict = {
        "geneProductId": uid,
        "limit": max(1, min(int(max_results), 100)),
    }
    if aspects:
        params["aspect"] = aspects
    try:
        data = await query_rest_api(
            f"{QUICKGO_BASE}/annotation/search",
            params=params,
            max_retries=1,
        )
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code if exc.response is not None else 0
        if code >= 500:
            logger.warning("QuickGO service error for %s: HTTP %d", uid, code)
            return f"QuickGO service temporarily unavailable (HTTP {code}) for {uid}."
        logger.exception("QuickGO query failed")
        return f"QuickGO query failed for {uid}: {exc}"
    except Exception as exc:
        logger.exception("QuickGO query failed")
        return f"QuickGO query failed for {uid}: {exc}"
    rows = []
    seen: set[tuple[str, str]] = set()
    for r in data.get("results", []) or []:
        gid = r.get("goId")
        ev = r.get("goEvidence") or ""
        if not gid:
            continue
        key = (gid, ev)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "go_id": gid,
                "go_name": r.get("goName"),
                "aspect": r.get("goAspect"),
                "evidence": ev,
                "qualifier": r.get("qualifier"),
            }
        )

    # QuickGO's annotation endpoint frequently returns ``goName: null``.
    # Enrich by bulk-fetching term labels from the ontology endpoint so the
    # downstream LLM doesn't hallucinate GO term names from the IDs.
    missing_ids = sorted({r["go_id"] for r in rows if not r["go_name"]})
    if missing_ids:
        # Cap the URL length: chunk into 50-id batches.
        name_map: dict[str, str] = {}
        for i in range(0, len(missing_ids), 50):
            chunk = missing_ids[i : i + 50]
            try:
                term_data = await query_rest_api(
                    f"{QUICKGO_BASE}/ontology/go/terms/{','.join(chunk)}",
                    max_retries=2,
                )
            except Exception as exc:
                logger.warning("QuickGO term-name enrichment failed: %s", exc)
                continue
            for t in (term_data or {}).get("results", []) or []:
                tid = t.get("id")
                tname = t.get("name")
                if tid and tname:
                    name_map[tid] = tname
        for r in rows:
            if not r["go_name"] and r["go_id"] in name_map:
                r["go_name"] = name_map[r["go_id"]]

    return json.dumps(
        {"uniprot": uid, "annotations": rows, "count": len(rows)},
        ensure_ascii=False,
        indent=2,
    )
