"""
Ingestion Routes - Endpoints for adding data to the knowledge base.

Supports:
- URL scraping (web pages)
- Text input
- File upload (PDF, DOCX, TXT)
"""
import logging
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends

from ..models import URLIngestRequest, TextIngestRequest, IngestResponse
from ...data import WebScraper, TextPreprocessor, TextChunker
from ...rag import RAGChain

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])


def get_rag_chain():
    """Dependency to get RAG chain instance."""
    from ..main import rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized. Check your API key.")
    return rag_chain


@router.post("/url", response_model=IngestResponse)
async def ingest_from_urls(
    request: URLIngestRequest,
    rag: RAGChain = Depends(get_rag_chain)
):
    """
    Ingest content from web URLs.
    
    Scrapes the provided URLs and adds the content to the knowledge base.
    """
    try:
        # Scrape URLs
        scraper = WebScraper()
        documents = scraper.scrape_urls(request.urls)
        
        if not documents:
            return IngestResponse(
                success=False,
                message="No content could be scraped from the provided URLs",
                documents_added=0
            )
        
        # Preprocess
        preprocessor = TextPreprocessor()
        cleaned_docs = preprocessor.clean_documents([d.to_dict() for d in documents])
        
        # Chunk
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.chunk_documents(cleaned_docs)
        
        # Add to vector store
        chunk_dicts = [c.to_dict() for c in chunks]
        rag.vector_store.add_chunks(chunk_dicts)
        
        return IngestResponse(
            success=True,
            message=f"Successfully ingested {len(documents)} URLs",
            documents_added=len(documents),
            chunks_created=len(chunks)
        )
        
    except Exception as e:
        logger.error(f"Error ingesting URLs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text", response_model=IngestResponse)
async def ingest_text(
    request: TextIngestRequest,
    rag: RAGChain = Depends(get_rag_chain)
):
    """
    Ingest raw text content.
    
    Adds the provided text directly to the knowledge base.
    """
    try:
        # Preprocess
        preprocessor = TextPreprocessor()
        cleaned_text = preprocessor.clean_text(request.text)
        
        if len(cleaned_text) < 50:
            return IngestResponse(
                success=False,
                message="Text too short after cleaning (minimum 50 characters)",
                documents_added=0
            )
        
        # Chunk
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.chunk_text(cleaned_text, source=request.source)
        
        # Add to vector store
        chunk_dicts = [c.to_dict() for c in chunks]
        rag.vector_store.add_chunks(chunk_dicts)
        
        return IngestResponse(
            success=True,
            message="Successfully ingested text",
            documents_added=1,
            chunks_created=len(chunks)
        )
        
    except Exception as e:
        logger.error(f"Error ingesting text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    rag: RAGChain = Depends(get_rag_chain)
):
    """
    Ingest content from uploaded file.
    
    Supports: .txt, .pdf, .docx
    """
    try:
        # Check file type
        filename = file.filename or "unknown"
        ext = filename.lower().split(".")[-1]
        
        if ext not in ["txt", "pdf", "docx"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Supported: txt, pdf, docx"
            )
        
        # Read content
        content = await file.read()
        
        # Extract text based on file type
        if ext == "txt":
            text = content.decode("utf-8", errors="ignore")
        elif ext == "pdf":
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext == "docx":
            from docx import Document
            import io
            doc = Document(io.BytesIO(content))
            text = "\n".join(para.text for para in doc.paragraphs)
        else:
            text = ""
        
        if not text.strip():
            return IngestResponse(
                success=False,
                message="No text content could be extracted from the file",
                documents_added=0
            )
        
        # Preprocess
        preprocessor = TextPreprocessor()
        cleaned_text = preprocessor.clean_text(text)
        
        # Chunk
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.chunk_text(cleaned_text, source=filename)
        
        # Add to vector store
        chunk_dicts = [c.to_dict() for c in chunks]
        rag.vector_store.add_chunks(chunk_dicts)
        
        return IngestResponse(
            success=True,
            message=f"Successfully ingested file: {filename}",
            documents_added=1,
            chunks_created=len(chunks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_knowledge_base(rag: RAGChain = Depends(get_rag_chain)):
    """
    Clear all documents from the knowledge base.
    
    WARNING: This cannot be undone!
    """
    try:
        rag.vector_store.delete_collection()
        return {"success": True, "message": "Knowledge base cleared"}
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ecommerce-data", response_model=IngestResponse)
async def load_ecommerce_data(rag: RAGChain = Depends(get_rag_chain)):
    """
    Load sample e-commerce product data into the knowledge base.
    Also builds the knowledge graph for entity relationships.
    
    Includes products from:
    - Shoes, Clothing, Electronics
    - Kitchen, Home Decor, Furniture
    - Grocery, Personal Care, Books
    - Baby Products, Pet Supplies, Garden
    """
    try:
        from ...data.ecommerce_data import get_all_products_as_documents, generate_all_products
        from ...knowledge_graph.graph import get_knowledge_graph, reset_knowledge_graph
        
        # Get sample product documents
        documents = get_all_products_as_documents()
        
        if not documents:
            return IngestResponse(
                success=False,
                message="No product data available",
                documents_added=0
            )
        
        # Add to vector store
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        rag.vector_store.add_documents(texts, metadatas=metadatas)
        
        # Build knowledge graph
        logger.info("Building knowledge graph...")
        reset_knowledge_graph()  # Clear existing graph
        kg = get_knowledge_graph()
        products = generate_all_products()
        kg.build_from_products(products)
        
        logger.info(f"Loaded {len(documents)} e-commerce product documents")
        
        return IngestResponse(
            success=True,
            message=f"Successfully loaded {len(documents)} products and built knowledge graph with {len(kg.entities)} entities",
            documents_added=len(documents),
            chunks_created=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Error loading e-commerce data: {e}")
        raise HTTPException(status_code=500, detail=str(e))



