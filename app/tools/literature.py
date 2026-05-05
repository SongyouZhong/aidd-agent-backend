"""Literature search tools (CORE, always loaded).

Adapted from Biomni's literature module but with three differences:
  * Pydantic-pruned to a fixed set of scientific fields.
  * Hard-capped output via ``@guarded_tool`` (≤3000 tokens, AC §2.2).
  * Sync libs (``pymed``, ``arxiv``) wrapped in ``asyncio.to_thread`` so
    they don't block the FastAPI event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.tools import tool

from app.tools.base import query_rest_api
from app.tools.preprocess import MAX_TOOL_TOKENS, guarded_tool
from app.tools.schemas import Paper

logger = logging.getLogger(__name__)


# --- internal helpers --------------------------------------------------

def _format_papers(papers: list[Paper]) -> str:
    if not papers:
        return "No papers found."
    return "\n\n".join(p.to_markdown() for p in papers)


async def _pubmed_query_async(query: str, max_papers: int) -> list[Paper]:
    """Fetch PubMed papers via NCBI E-utilities (no pymed dependency)."""
    EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    search_data = await query_rest_api(
        f"{EUTILS}/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_papers,
        },
    )
    id_list: list[str] = (search_data.get("esearchresult") or {}).get("idlist") or []
    if not id_list:
        return []
    summary_data = await query_rest_api(
        f"{EUTILS}/esummary.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json",
        },
    )
    result_map: dict = summary_data.get("result") or {}
    papers: list[Paper] = []
    for pmid in id_list:
        item = result_map.get(pmid)
        if not item or not isinstance(item, dict):
            continue
        doi: str | None = None
        for aid in item.get("articleids") or []:
            if aid.get("idtype") == "doi":
                doi = aid.get("value") or None
                break
        year: int | None = None
        for date_field in ("pubdate", "epubdate", "sortpubdate"):
            date_str = item.get(date_field) or ""
            if date_str:
                try:
                    year = int(date_str[:4])
                    break
                except (ValueError, TypeError):
                    pass
        papers.append(
            Paper(
                title=(item.get("title") or "").strip(),
                journal=item.get("fulljournalname") or item.get("source") or None,
                year=year,
                doi=doi,
                pmid=pmid,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            )
        )
    return papers


def _arxiv_to_paper(item: Any) -> Paper:
    return Paper(
        title=(item.title or "").strip(),
        abstract=(item.summary or "").strip() or None,
        authors=[a.name for a in getattr(item, "authors", [])],
        year=item.published.year if getattr(item, "published", None) else None,
        url=getattr(item, "entry_id", None),
    )


# --- sync workhorses (run in a worker thread) -------------------------

def _arxiv_query_sync(query: str, max_papers: int) -> list[Paper]:
    import arxiv  # type: ignore

    client = arxiv.Client()
    search = arxiv.Search(
        query=query, max_results=max_papers, sort_by=arxiv.SortCriterion.Relevance
    )
    return [_arxiv_to_paper(p) for p in client.results(search)]


# --- public tools ------------------------------------------------------

@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_pubmed(query: str, max_papers: int = 5) -> str:
    """Search PubMed for biomedical literature.

    Args:
        query: Free-form search query, e.g. "EGFR T790M resistance".
        max_papers: Maximum number of papers to return (1-20).

    Returns a markdown summary with title, journal, year, PMID/DOI,
    and abstract for each result. Always cite PMID/DOI in downstream
    answers (anti-hallucination policy).
    """
    max_papers = max(1, min(int(max_papers), 20))
    try:
        papers = await _pubmed_query_async(query, max_papers)
    except Exception as exc:
        logger.exception("PubMed query failed")
        return f"PubMed query failed: {exc}"
    return _format_papers(papers)


@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_arxiv(query: str, max_papers: int = 5) -> str:
    """Search arXiv for preprints (biology / chemistry / medicine).

    Args:
        query: Free-form search query.
        max_papers: Maximum number of papers (1-20).
    """
    max_papers = max(1, min(int(max_papers), 20))
    try:
        papers = await asyncio.to_thread(_arxiv_query_sync, query, max_papers)
    except Exception as exc:
        logger.exception("arXiv query failed")
        return f"arXiv query failed: {exc}"
    return _format_papers(papers)
