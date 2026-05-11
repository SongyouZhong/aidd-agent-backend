"""Pathway-related deferred tools: KEGG / Reactome / STRING."""

from __future__ import annotations

import json
import logging

import httpx

from langchain_core.tools import tool

from app.tools.base import query_rest_api
from app.tools.preprocess import MAX_TOOL_TOKENS, guarded_tool

logger = logging.getLogger(__name__)


# --- KEGG -------------------------------------------------------------

KEGG_BASE = "https://rest.kegg.jp"


def _parse_kegg_link(text: str, value_field: str = "pathway_id") -> list[dict]:
    """KEGG link/get returns TSV. We turn it into a list of dicts."""
    rows = []
    for line in (text or "").splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            rows.append({"gene": parts[0], value_field: parts[1]})
    return rows


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_kegg(gene_id: str, organism: str = "hsa") -> str:
    """List KEGG pathways containing a given gene.

    Args:
        gene_id: KEGG gene id (``hsa:1956``) **or** an Entrez gene id
                 (``1956``) — the latter is auto-converted using the
                 organism prefix.
        organism: KEGG organism code (default ``hsa`` = human).
    """
    g = gene_id.strip()
    if ":" not in g:
        g = f"{organism}:{g}"

    # 1) link gene -> pathway ids
    try:
        link_text = await query_rest_api(
            f"{KEGG_BASE}/link/pathway/{g}",
            expect_json=False,
        )
    except Exception as exc:
        logger.exception("KEGG link failed")
        return f"KEGG link failed for {g}: {exc}"

    pathway_rows = _parse_kegg_link(link_text)
    if not pathway_rows:
        return f"No KEGG pathways found for {g}."

    # 2) Fetch human-readable names for each pathway via /list/pathway
    try:
        list_text = await query_rest_api(
            f"{KEGG_BASE}/list/pathway/{organism}",
            expect_json=False,
        )
    except Exception as exc:
        logger.warning("KEGG pathway name fetch failed: %s", exc)
        list_text = ""

    name_map: dict[str, str] = {}
    for line in list_text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2:
            pid, pname = parts
            # KEGG returns ``path:hsa00010`` in /link but ``hsa00010`` in /list.
            name_map[pid] = pname
            name_map[f"path:{pid}"] = pname

    out = []
    for r in pathway_rows[:50]:
        pid = r["pathway_id"]
        out.append(
            {
                "pathway_id": pid,
                "name": name_map.get(pid),
                "url": f"https://www.kegg.jp/pathway/{pid.replace('path:', '')}",
            }
        )
    return json.dumps(
        {"gene": g, "pathways": out, "count": len(out)},
        ensure_ascii=False,
        indent=2,
    )


# --- Reactome ---------------------------------------------------------

REACTOME_CONTENT = "https://reactome.org/ContentService"


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_reactome(uniprot_id: str, species: str = "Homo sapiens") -> str:
    """Find Reactome pathways for a UniProt protein.

    Args:
        uniprot_id: UniProt accession (e.g. ``P00533``).
        species: Species filter for client-side post-filtering (default
            ``Homo sapiens``). Note: the underlying Reactome ContentService
            endpoint ``/data/mapping/UniProt/{acc}/pathways`` does NOT accept
            a ``species`` query parameter — passing one returns 404. We fetch
            unfiltered then filter in-process.
    """
    uid = uniprot_id.strip().upper()
    url = f"{REACTOME_CONTENT}/data/mapping/UniProt/{uid}/pathways"
    try:
        # Reactome's ContentService is intermittently flaky; allow more retries
        # so a single 5xx burst doesn't blank out the pathway node.
        # IMPORTANT: do NOT pass `species` as a query param — that endpoint
        # rejects it with 404. We filter client-side below.
        data = await query_rest_api(url, max_retries=4)
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code if exc.response is not None else 0
        if code < 500:
            # 404 from this endpoint = "no Reactome mapping for this UniProt".
            # Return a terminal message so the LLM does not retry.
            return (
                f"Reactome: no pathways mapped to {uid} (HTTP {code}). "
                "This is a terminal result — do NOT call query_reactome again "
                "for the same accession; treat as 'no Reactome data'."
            )
        logger.warning("Reactome server error for %s: HTTP %d", uid, code)
        # Structured error so downstream LLM/synthesize can distinguish
        # "transiently unavailable" from "no data exists".
        return json.dumps(
            {"source": "Reactome", "uniprot": uid, "error": f"HTTP {code}", "retryable": True},
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.exception("Reactome query failed")
        return json.dumps(
            {"source": "Reactome", "uniprot": uid, "error": str(exc), "retryable": True},
            ensure_ascii=False,
        )

    if not isinstance(data, list):
        return f"No Reactome pathways found for {uid}."
    species_norm = (species or "").strip().lower()
    rows = []
    for p in data:
        sp_field = p.get("species")
        sp_name = (
            sp_field.get("displayName") if isinstance(sp_field, dict) else None
        )
        if species_norm and sp_name and sp_name.lower() != species_norm:
            continue
        st_id = p.get("stId")
        rows.append(
            {
                "stable_id": st_id,
                "name": p.get("displayName"),
                "species": sp_name,
                "url": f"https://reactome.org/PathwayBrowser/#/{st_id}" if st_id else None,
            }
        )
    return json.dumps(
        {"uniprot": uid, "pathways": rows, "count": len(rows)},
        ensure_ascii=False,
        indent=2,
    )


# --- STRING-DB --------------------------------------------------------

STRING_BASE = "https://version-12-0.string-db.org/api"


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_stringdb(
    identifiers: str,
    species: int = 9606,
    required_score: int = 700,
    max_partners: int = 25,
) -> str:
    """Fetch protein-protein interaction network for one or more genes/proteins.

    Args:
        identifiers: Comma-separated identifiers (e.g. ``"EGFR,ERBB2"``
                     or a single UniProt accession).
        species: NCBI taxonomy id (default 9606 = human).
        required_score: STRING combined-score threshold (0-1000, default 700).
        max_partners: Maximum number of partners to return per query node.
    """
    ids = identifiers.strip().replace(" ", "")
    params = {
        "identifiers": ids,
        "species": species,
        "required_score": required_score,
        "limit": max(1, min(int(max_partners), 100)),
    }
    try:
        data = await query_rest_api(
            f"{STRING_BASE}/json/network",
            params=params,
        )
    except Exception as exc:
        logger.exception("STRING network query failed")
        return f"STRING query failed: {exc}"
    rows = []
    if isinstance(data, list):
        for r in data:
            rows.append(
                {
                    "a": r.get("preferredName_A"),
                    "b": r.get("preferredName_B"),
                    "score": r.get("score"),
                    "experiments": r.get("escore"),
                    "database": r.get("dscore"),
                }
            )
    return json.dumps(
        {
            "identifiers": ids,
            "species": species,
            "required_score": required_score,
            "edges": rows,
            "count": len(rows),
        },
        ensure_ascii=False,
        indent=2,
    )
