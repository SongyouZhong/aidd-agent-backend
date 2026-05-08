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

from langchain_core.prompts import PromptTemplate

# Custom Cypher Generation Prompt based on the actual WikiPathways schema
CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}

Note: 
- Genes/Proteins are typically labeled as 'GeneProduct', 'Protein', or 'Entity'. 
- The 'name' property contains the gene symbol (e.g., 'TARDBP').
- Pathways are labeled as 'Pathway' with a 'name' property.
- Relationship 'CONTAINS' connects Pathway to GeneProduct/Protein/Entity.
- Relationship 'ACTIVATES', 'INHIBITS', 'INTERACTS_WITH' connects entities.

Examples:
- "What pathways involve the gene TARDBP?"
  MATCH (p:Pathway)-[:CONTAINS]->(g) WHERE g.name = 'TARDBP' RETURN p.name
- "What genes interact with TARDBP in the Apoptosis pathway?"
  MATCH (p:Pathway {{name: 'Apoptosis'}})-[:CONTAINS]->(g1 {{name: 'TARDBP'}})-[:INTERACTS_WITH]-(g2) RETURN g2.name

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

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
    chain = get_cypher_chain()
    if not chain:
        return "Error: Could not connect to the WikiPathways Neo4j database."

    try:
        result = chain.invoke({"query": query})
        return result.get("result", "No result found.")
    except Exception as e:
        logger.exception("GraphQA chain failed")
        return f"Failed to query the knowledge graph: {e}"
