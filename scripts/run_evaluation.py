import os
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import asyncio
from dotenv import load_dotenv
from src.vectorstore.store import VectorStore
from src.rag.generator import LLMGenerator
from src.rag.chain import RAGChain
from src.evaluation.ablation import run_ablation_study
from src.knowledge_graph.graph import get_knowledge_graph, reset_knowledge_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_kg_from_vs(vs):
    """Rebuilds KG from vector store documents."""
    logger.info("Rebuilding Knowledge Graph...")
    kg = get_knowledge_graph()
    reset_knowledge_graph()
    kg = get_knowledge_graph()
    
    all_docs = vs.get_all_documents()
    products = []
    
    for doc in all_docs:
        metadata = doc.get("metadata", {})
        name = metadata.get("title") or metadata.get("name")
        if name and metadata.get("brand"):
            products.append({
                "name": name,
                "brand": metadata.get("brand"),
                "category": metadata.get("category", ""),
                "price": metadata.get("price", 0),
                "rating": metadata.get("rating"),
                "reviews_count": metadata.get("reviews_count"),
                "description": metadata.get("description", ""),
                "features": metadata.get("features", [])
            })
            
    kg.build_from_products(products)
    stats = kg.get_stats()
    logger.info(f"KG Built: {stats}")

def main():
    print("--- STARTING FULL EVALUATION ---")
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found.")
        return

    # 1. Init Vector Store
    vs = VectorStore(persist_directory="./data/vector_db")
    
    # 2. Build KG
    build_kg_from_vs(vs)
    
    # 3. Init RAG
    gen = LLMGenerator(api_key=api_key)
    chain = RAGChain(vs, gen, top_k=3)
    
    # 4. Run Study
    print("Running Ablation Study... This may take 5-10 minutes.")
    results = run_ablation_study(
        rag_chain=chain, 
        llm_generator=gen,
        output_dir="./data/evaluation_results"
    )
    
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(results.to_markdown_report())
    print("="*50)
    print(f"Detailed JSON results saved to: {results.config['output_dir']}")

if __name__ == "__main__":
    main()
