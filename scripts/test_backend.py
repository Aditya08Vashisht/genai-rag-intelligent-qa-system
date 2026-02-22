import sys
import os
import logging

# Add project root to path
sys.path.append(os.getcwd())

from src.rag.chain import RAGChain
from src.vectorstore.store import VectorStore
from src.rag.generator import LLMGenerator
from src.knowledge_graph.graph import get_knowledge_graph, reset_knowledge_graph
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_backend():
    print("--- BACKEND VALIDATION TEST ---")
    load_dotenv()
    
    # 1. Load Vector Store
    logger.info("Loading Vector Store...")
    vs = VectorStore(persist_directory="./data/vector_db")
    count = vs.count()
    print(f"Vector Store Documents: {count}")
    
    if count == 0:
        print("ERROR: Vector Store is empty! Run repopulate_db.py first.")
        return

    # 2. Rebuild Knowledge Graph (Simulating main.py startup)
    logger.info("Building Knowledge Graph from Vector Store...")
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
    print(f"KG Stats: {stats['total_entities']} entities, {stats['total_relationships']} rels")
    
    # 3. Init RAG
    print("Initializing RAG Chain...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found!")
        return
        
    gen = LLMGenerator(api_key=api_key)
    chain = RAGChain(vs, gen, top_k=3)
    
    # 4. Test Queries
    # We want to check if "Apple" query returns "iPhone" and NOT "Redmi"
    test_queries = [
        "What products does Samsung make?",
        "Does Apple make a Redmi phone?",
        "List 3 Nike shoe models."
    ]
    
    for q in test_queries:
        print(f"\nQUERY: {q}")
        result = chain.query(q, retrieval_mode="hybrid")
        print(f"ANSWER: {result['answer']}")
        print("-" * 50)
        
        # Validation Logic
        ans = result['answer'].lower()
        if "Samsung" in q:
            if "galaxy" in ans and "iphone" not in ans:
                print("✅ PASS: Samsung -> Galaxy")
            elif "iphone" in ans:
                print("❌ FAIL: Found iPhone in Samsung query")
            else:
                print("⚠️ WARN: Galaxy not found?")
                
        if "Apple" in q and "Redmi" in q:
            if "no" in ans or "does not" in ans:
                print("✅ PASS: Correctly denied Apple-Redmi connection")
            else:
                print("❌ FAIL: Did not deny Apple-Redmi")

if __name__ == "__main__":
    test_backend()
