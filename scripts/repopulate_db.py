import shutil
from pathlib import Path
import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.data.ecommerce_data import get_all_products_as_documents
from src.vectorstore.store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def repopulate_db():
    # 1. Clear existing DB
    db_path = Path("./data/vector_db")
    if db_path.exists():
        logger.warning(f"Removing existing vector DB at {db_path}")
        shutil.rmtree(db_path)
    
    # 2. Initialize new internal DB
    logger.info("Initializing new Vector Store...")
    vs = VectorStore(persist_directory="./data/vector_db")
    
    # 3. Generate Data
    logger.info("Generating 4000+ products with STRICT BRAND LOGIC...")
    docs = get_all_products_as_documents(4000)
    
    # 4. Ingest
    logger.info(f"Ingesting {len(docs)} documents into Vector Store...")
    
    contents = [d["content"] for d in docs]
    metadatas = [d["metadata"] for d in docs]
    
    vs.add_documents(contents, metadatas=metadatas)
    
    logger.info("Database repopulation COMPLETE.")
    logger.info(f"Total documents: {vs.count()}")

if __name__ == "__main__":
    repopulate_db()
