import os
import sys
import asyncio
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.tools.graph_rag import query_wikipathways_graph

async def main():
    load_dotenv()
    print("Testing GraphRAG query...")
    query = "What genes are involved in Apoptosis?"
    
    # query_wikipathways_graph is a LangChain @tool, we can invoke it.
    result = query_wikipathways_graph.invoke({"query": query})
    print(f"Result:\n{result}")

if __name__ == "__main__":
    asyncio.run(main())
