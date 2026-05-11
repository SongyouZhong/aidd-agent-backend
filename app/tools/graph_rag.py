"""WikiPathways GraphRAG Tool."""

import os
import logging
from contextvars import ContextVar
from langchain_core.tools import tool
from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

from app.agent.llm_provider import get_graph_rag_llm

logger = logging.getLogger(__name__)

# Lazy initialization of the graph chain to avoid connection errors on startup if Neo4j is down
_cypher_chain = None

# Per-task counter of consecutive empty/unknown results.
# ContextVar is propagated across asyncio tasks within the same node, so each
# pathway-node run gets its own isolated counter (not a process-global one).
_empty_streak: ContextVar[int] = ContextVar("wikipathways_empty_streak", default=0)
_MAX_EMPTY_STREAK = 2  # after 2 consecutive empties, short-circuit further calls

# Hard cap on total GraphRAG calls per task (regardless of result content).
# Each call costs 10-20s of Cypher-LLM + Cypher execution, so 3 calls is
# already half of the pathway node's 180s budget.
_total_calls: ContextVar[int] = ContextVar("wikipathways_total_calls", default=0)
_MAX_TOTAL_CALLS = 3

# Track previously seen results within the same task. A repeated identical
# (or near-identical) answer means the LLM is asking variant questions that
# the graph cannot distinguish — also count toward the empty-streak so we
# short-circuit instead of looping.
_seen_results: ContextVar[set[str] | None] = ContextVar(
    "wikipathways_seen_results", default=None
)

# Custom Cypher Generation Prompt based on the actual WikiPathways schema
CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}

CRITICAL SCHEMA NOTES:
- All biological entities (genes, proteins, metabolites) are labeled :Entity.
  They may also carry an additional label from their GPML Type (e.g. :GeneProduct, :Protein, :Metabolite).
- The 'name' property stores the DISPLAY LABEL from WikiPathways (e.g. 'TDP-43', 'p53', 'EGFR').
  It does NOT store gene symbols like 'TARDBP' or 'TP53'. Gene symbols are usually in db_id (Entrez) or db_name (Ensembl/HGNC).
- Pathways are labeled :Pathway with properties: name, id (WP identifier), organism.
- Relationships: (Pathway)-[:CONTAINS]->(Entity), (Entity)-[:ACTIVATES|INHIBITS|INTERACTS_WITH]->(Entity).
- ALWAYS search by display name, not gene symbol.
  For a target like TDP-43 (gene TARDBP), use: e.name = 'TDP-43'
  When uncertain of the exact display name, use case-insensitive regex:
    WHERE e.name =~ '(?i).*TDP.43.*' OR e.name =~ '(?i).*TARDBP.*'

❗ CYPHER LABEL RULES — read carefully:
- A node can only be matched by ONE label per pattern position. To match
  any of multiple labels, use a `WHERE` clause with the `:Label` predicate
  combined by `OR`. NEVER stuff multiple label names inside a single
  backtick-quoted label literal.
- ❌ WRONG (creates a fake label literally named "GeneProduct OR g"):
    MATCH (g) WHERE g.name = 'TARDBP' AND (g:`GeneProduct OR g`:Protein OR g:Entity)
- ✅ CORRECT — use a generic match on :Entity (the parent label) and rely
   on properties; this covers GeneProduct/Protein/Metabolite at once:
    MATCH (p:Pathway)-[:CONTAINS]->(e:Entity)
    WHERE e.name =~ '(?i).*TDP.43.*'
    RETURN DISTINCT p.name
- ✅ CORRECT — explicit OR over labels:
    MATCH (p:Pathway)-[:CONTAINS]->(e)
    WHERE (e:GeneProduct OR e:Protein) AND e.name = 'TDP-43'
    RETURN DISTINCT p.name

Examples:
- "What pathways involve TDP-43?"
  MATCH (p:Pathway)-[:CONTAINS]->(e:Entity)
  WHERE e.name =~ '(?i).*TDP.43.*' OR e.name =~ '(?i).*TARDBP.*'
  RETURN DISTINCT p.name

- "What genes interact with TDP-43?"
  MATCH (e1:Entity)-[:INTERACTS_WITH|ACTIVATES|INHIBITS]-(e2:Entity)
  WHERE e1.name =~ '(?i).*TDP.43.*'
  RETURN DISTINCT e2.name

- "What pathways involve EGFR?"
  MATCH (p:Pathway)-[:CONTAINS]->(e:Entity)
  WHERE e.name =~ '(?i).*EGFR.*'
  RETURN DISTINCT p.name

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

def reset_cypher_chain():
    """Force recreation of the cached chain (e.g. after prompt update)."""
    global _cypher_chain
    _cypher_chain = None

def get_cypher_chain():
    global _cypher_chain
    if _cypher_chain is not None:
        return _cypher_chain

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "aidd_neo4j_password")

    try:
        graph = Neo4jGraph(
            url=neo4j_uri, 
            username=neo4j_user, 
            password=neo4j_password
        )
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        return None

    try:
        llm = get_graph_rag_llm()
    except RuntimeError as e:
        logger.error("GraphRAG LLM initialisation failed: %s", e)
        return None

    _cypher_chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm,
        qa_llm=llm,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        top_k=20
    )
    return _cypher_chain

@tool
def query_graph_schema() -> str:
    """Get the schema of the WikiPathways graph database, including node labels, properties, and relationships.
    Use this tool when you are unsure about the graph structure or what properties to use for filtering.
    """
    chain = get_cypher_chain()
    if not chain:
        return "Error: Could not connect to the WikiPathways Neo4j database."
    try:
        return chain.graph.get_schema
    except Exception as e:
        logger.exception("Failed to get graph schema")
        return f"Failed to retrieve schema: {e}"

@tool
def query_wikipathways_graph(query: str) -> str:
    """Use this tool to answer complex questions about biological pathways, genes, and interactions.
    
    This tool uses GraphRAG (Neo4j) to query the WikiPathways database.
    Pass a clear, natural language question describing what you want to find.
    Example: 'What genes are contained in the Apoptosis pathway?'
    """
    # Short-circuit: if the previous calls in this same task already returned
    # empty/unknown twice in a row, don't burn another 10-20s LLM round
    # generating yet another Cypher that will likely also miss.
    if _empty_streak.get() >= _MAX_EMPTY_STREAK:
        return (
            "WikiPathways graph has no matching data for this target "
            "(2 consecutive empty/duplicate results). Stop calling "
            "query_wikipathways_graph and rely on KEGG/Reactome/STRING results "
            "instead."
        )
    # Hard cap: regardless of result content, never exceed N calls per task.
    if _total_calls.get() >= _MAX_TOTAL_CALLS:
        return (
            f"WikiPathways GraphRAG hard cap reached ({_MAX_TOTAL_CALLS} calls per node). "
            "Stop calling query_wikipathways_graph and synthesize from existing results."
        )
    _total_calls.set(_total_calls.get() + 1)

    chain = get_cypher_chain()
    if not chain:
        return "Error: Could not connect to the WikiPathways Neo4j database."

    try:
        result = chain.invoke({"query": query})
        answer = result.get("result", "No result found.")
    except Exception as e:
        logger.exception("GraphQA chain failed")
        return f"Failed to query the knowledge graph: {e}"

    # Detect "empty" responses: GraphCypherQAChain returns either
    # "I don't know the answer." (no Cypher results) or a short stub.
    norm = (answer or "").strip().lower()
    is_empty = (
        not norm
        or "don't know" in norm
        or "do not know" in norm
        or norm == "no result found."
    )
    # Also treat duplicate (already-seen) answers as a wasted call.
    seen = _seen_results.get()
    if seen is None:
        seen = set()
        _seen_results.set(seen)
    is_duplicate = (not is_empty) and (norm in seen)
    seen.add(norm)

    if is_empty or is_duplicate:
        _empty_streak.set(_empty_streak.get() + 1)
    else:
        _empty_streak.set(0)
    if is_duplicate:
        return (
            f"{answer}\n\n[NOTE: identical to a previous GraphRAG result in this node — "
            "do not query the graph again with paraphrased questions.]"
        )
    return answer
