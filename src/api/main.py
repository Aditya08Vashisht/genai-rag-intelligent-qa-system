"""
FastAPI Application - Main entry point.

GenAI RAG Intelligent Q&A System
Using Google Gemini (FREE tier)
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from dotenv import load_dotenv

from .models import HealthResponse
from .routes import ingest, query, evaluation
from ..vectorstore import VectorStore, EmbeddingModel
from ..rag import RAGChain, LLMGenerator
from ..knowledge_graph import get_knowledge_graph

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global RAG chain instance
rag_chain: Optional[RAGChain] = None


def initialize_rag():
    """Initialize the RAG chain with Gemini LLM."""
    global rag_chain
    
    try:
        # Get API key
        google_api_key = os.getenv("GOOGLE_API_KEY", "")
        
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env file!")
        
        # Initialize embedding model (FREE - local HuggingFace)
        embedding_model = EmbeddingModel(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        
        # Initialize vector store (FREE - local storage)
        vector_store = VectorStore(
            collection_name="documents",
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./data/vector_db"),
            embedding_model=embedding_model
        )
        
        # Initialize Gemini LLM generator
        generator = LLMGenerator(
            api_key=google_api_key,
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        )
        logger.info("Using Gemini LLM (gemini-2.0-flash)")
        
        # Create RAG chain with lower threshold
        rag_chain = RAGChain(
            vector_store=vector_store,
            generator=generator,
            top_k=5,
            score_threshold=0.1  # Lowered from 0.3
        )
        logger.info("RAG chain initialized successfully")
        
        # Build Knowledge Graph from existing documents
        try:
            doc_count = vector_store.count()
            if doc_count > 0:
                logger.info(f"Building Knowledge Graph from {doc_count} documents...")
                kg = get_knowledge_graph()
                
                # Extract product data from stored documents
                all_docs = vector_store.get_all_documents()
                products = []
                for doc in all_docs:
                    metadata = doc.get("metadata", {})
                    # Check for 'title' (product documents) or 'name'
                    name = metadata.get("title") or metadata.get("name")
                    if name and metadata.get("brand"):  # Only process product documents
                        products.append({
                            "name": name,
                            "brand": metadata.get("brand"),
                            "category": metadata.get("category", ""),
                            "price": metadata.get("price", 0),
                            "rating": metadata.get("rating"),
                            "reviews_count": metadata.get("reviews_count"),
                            "description": metadata.get("description", ""),
                            "features": metadata.get("features", "").split(", ") if isinstance(metadata.get("features"), str) else metadata.get("features", [])
                        })
                
                if products:
                    kg.build_from_products(products)
                    stats = kg.get_stats()
                    logger.info(f"Knowledge Graph built: {stats['total_entities']} entities, {stats['total_relationships']} relationships")
                else:
                    logger.info("No product documents found for Knowledge Graph")
        except Exception as kg_error:
            logger.warning(f"Could not build Knowledge Graph on startup: {kg_error}")
            
    except Exception as e:
        logger.error(f"Error initializing RAG chain: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    logger.info("Starting GenAI RAG Q&A System...")
    initialize_rag()
    yield
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="GenAI RAG Intelligent Q&A System",
    description="""
    ## A Production-Grade RAG System
    
    This API allows you to:
    - **Ingest** data from web pages, text, or files
    - **Query** the knowledge base with natural language questions
    - **Get answers** grounded in your data with source citations
    
    ### Components:
    - 🧠 **Embeddings**: HuggingFace sentence-transformers (local, FREE)
    - 💾 **Vector Store**: NumPy-based (local, FREE)
    - 🤖 **LLM**: Google Gemini (FREE tier)
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(evaluation.router)

# Frontend path
frontend_path = Path(__file__).resolve().parent.parent.parent / "frontend"
logger.info(f"Frontend path: {frontend_path}")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend."""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return {"message": "GenAI RAG Q&A System API", "docs": "/docs"}


@app.get("/styles.css", include_in_schema=False)
async def get_styles():
    """Serve CSS file."""
    css_path = frontend_path / "styles.css"
    if css_path.exists():
        return FileResponse(str(css_path), media_type="text/css")
    return {"error": "CSS not found"}


@app.get("/app.js", include_in_schema=False)
async def get_js():
    """Serve JS file."""
    js_path = frontend_path / "app.js"
    if js_path.exists():
        return FileResponse(str(js_path), media_type="application/javascript")
    return {"error": "JS not found"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
