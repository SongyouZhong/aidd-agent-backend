"""Semantic Scholar literature search and graph traversal tools."""

import logging
from typing import Any

from langchain_core.tools import tool

from app.core.config import settings
from app.tools.base import query_rest_api
from app.tools.preprocess import MAX_TOOL_TOKENS, guarded_tool

logger = logging.getLogger(__name__)

# Base API endpoint
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"

# The specific fields requested by the user
PAPER_FIELDS = "paperId,title,abstract,year,authors,url,citationCount"

def _get_headers() -> dict[str, str]:
    headers = {}
    if settings.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY
    return headers

def _format_paper(p: dict[str, Any]) -> str:
    """Format a Semantic Scholar paper response to Markdown."""
    title = p.get("title") or "Untitled"
    year = p.get("year") or "Unknown Year"
    url = p.get("url") or f"https://www.semanticscholar.org/paper/{p.get('paperId')}"
    citation_count = p.get("citationCount", 0)
    
    authors_list = p.get("authors") or []
    authors_str = ", ".join([a.get("name") for a in authors_list if a.get("name")])
    if not authors_str:
        authors_str = "Unknown Authors"
        
    abstract = p.get("abstract") or "No abstract available."
    # Truncate extremely long abstracts slightly just in case
    if len(abstract) > 2000:
        abstract = abstract[:2000] + "..."

    return (
        f"### {title}\n"
        f"- **Authors:** {authors_str}\n"
        f"- **Year:** {year}\n"
        f"- **Citations:** {citation_count}\n"
        f"- **URL:** {url}\n"
        f"- **Paper ID:** {p.get('paperId')}\n"
        f"**Abstract:** {abstract}\n"
    )

@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_semantic_scholar_search(
    query: str,
    year: str = None,
    sort_by_citations: bool = False,
    max_results: int = 5,
    min_citations: int = 0,
) -> str:
    """Search for biomedical literature using Semantic Scholar.

    Args:
        query: Free-form search query, e.g. "TDP-43 Alzheimer's disease".
        year: Optional year filter, e.g. "2023-2025" or "2024".
        sort_by_citations: If True, fetches more results and sorts by citation count (descending).
        max_results: Maximum number of papers to return (1-20).
        min_citations: Minimum citation count threshold; papers below this value are excluded.
            If fewer qualifying papers exist, the tool returns all that qualify.

    Returns a markdown summary with title, authors, year, citation count, URL, and abstract.
    """
    max_results = max(1, min(int(max_results), 20))
    # Fetch a larger pool when filtering/sorting so we have enough candidates after pruning.
    limit = 100 if (sort_by_citations or min_citations > 0) else max_results

    params: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "fields": PAPER_FIELDS,
    }
    if year:
        params["year"] = str(year)

    try:
        data = await query_rest_api(
            f"{SEMANTIC_SCHOLAR_API_URL}/paper/search",
            params=params,
            headers=_get_headers()
        )
    except Exception as exc:
        logger.exception("Semantic Scholar search query failed")
        return f"Semantic Scholar search failed: {exc}"

    papers = data.get("data") or []
    if not papers:
        return "No papers found matching the query."

    # Filter by minimum citation count before sorting/truncating.
    if min_citations > 0:
        papers = [p for p in papers if int(p.get("citationCount") or 0) >= min_citations]
        if not papers:
            return f"No papers found with ≥{min_citations} citations matching the query."

    # Apply citation sorting locally if requested.
    if sort_by_citations:
        papers.sort(key=lambda x: int(x.get("citationCount") or 0), reverse=True)

    # Take the top max_results.
    papers = papers[:max_results]

    return "\n".join([_format_paper(p) for p in papers])

@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_semantic_scholar_paper(paper_id: str) -> str:
    """Get detailed information about a specific paper from Semantic Scholar using its ID.
    
    Args:
        paper_id: The Semantic Scholar paperId (e.g., returned from query_semantic_scholar_search).
        
    Returns detailed markdown about the paper including abstract and citation count.
    """
    try:
        data = await query_rest_api(
            f"{SEMANTIC_SCHOLAR_API_URL}/paper/{paper_id}",
            params={"fields": PAPER_FIELDS},
            headers=_get_headers()
        )
    except Exception as exc:
        logger.exception(f"Semantic Scholar paper details failed for {paper_id}")
        return f"Failed to retrieve paper details: {exc}"
        
    if not data or "title" not in data:
        return f"Paper ID {paper_id} not found."
        
    return _format_paper(data)

@tool
@guarded_tool(max_tokens=MAX_TOOL_TOKENS)
async def query_semantic_scholar_citations(paper_id: str, max_results: int = 10) -> str:
    """Find papers that have cited a specific paper (forward citations).
    
    Args:
        paper_id: The Semantic Scholar paperId of the original paper.
        max_results: Maximum number of citing papers to return (1-20).
        
    Returns a markdown list of the most influential citing papers.
    """
    max_results = max(1, min(int(max_results), 20))
    # We fetch a bit more to ensure we get ones with actual abstracts/titles, 
    # then return up to max_results.
    params = {
        "limit": 50,
        "fields": f"contexts,intents,isInfluential,citingPaper.{PAPER_FIELDS.replace(',', ',citingPaper.')}"
    }
    
    try:
        data = await query_rest_api(
            f"{SEMANTIC_SCHOLAR_API_URL}/paper/{paper_id}/citations",
            params=params,
            headers=_get_headers()
        )
    except Exception as exc:
        logger.exception(f"Semantic Scholar citations failed for {paper_id}")
        return f"Failed to retrieve citations: {exc}"
        
    citations = data.get("data") or []
    if not citations:
        return f"No citations found for paper {paper_id}."
        
    # Sort citations to prioritize "influential" citations, then by their own citation count
    def _score(cit: dict) -> tuple:
        is_influential = cit.get("isInfluential", False)
        citing_paper = cit.get("citingPaper") or {}
        citation_count = citing_paper.get("citationCount", 0)
        # Return tuple: (True/False, citationCount) so influential ones come first
        return (is_influential, int(citation_count or 0))
        
    citations.sort(key=_score, reverse=True)
    
    results = []
    for cit in citations:
        cp = cit.get("citingPaper")
        if not cp or not cp.get("title"):
            continue
            
        intent = cit.get("intents") or []
        intent_str = f" [Intent: {', '.join(intent)}]" if intent else ""
        influential_str = " 🔥 **Influential Citation**" if cit.get("isInfluential") else ""
        
        paper_md = _format_paper(cp)
        results.append(f"{paper_md.strip()}{influential_str}{intent_str}\n---")
        
        if len(results) >= max_results:
            break
            
    if not results:
        return "Found citations but none had complete metadata."
        
    return "\n\n".join(results)
