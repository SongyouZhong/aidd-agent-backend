"""WikiPathways GraphRAG Tool."""

import os
import logging
from langchain_core.tools import tool
from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# Lazy initialization of the graph chain to avoid connection errors on startup if Neo4j is down
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

    # Use Qwen for Cypher Generation
    llm = ChatOpenAI(
        model=settings.QWEN_MODEL,
        base_url=settings.QWEN_BASE_URL,
        api_key=settings.QWEN_API_KEY or "empty",
        temperature=0.0
    )

    _cypher_chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm,
        qa_llm=llm,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True,
        top_k=20
    )
    return _cypher_chain

@tool
def query_wikipathways_graph(query: str) -> str:
    """Use this tool to answer complex questions about biological pathways, genes, and interactions.
    
    This tool uses GraphRAG (Neo4j) to query the WikiPathways database.
    Pass a clear, natural language question describing what you want to find.
    Example: 'What genes are contained in the Apoptosis pathway?'
    """
    chain = get_cypher_chain()
    if not chain:
        return "Error: Could not connect to the WikiPathways Neo4j database."

    try:
        result = chain.invoke({"query": query})
        return result.get("result", "No result found.")
    except Exception as e:
        logger.exception("GraphQA chain failed")
        return f"Failed to query the knowledge graph: {e}"
