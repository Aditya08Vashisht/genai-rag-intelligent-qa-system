"""
Query Routes - Endpoints for querying the knowledge base.

Supports:
- Single question queries
- Chat with conversation history
"""
import logging
from fastapi import APIRouter, HTTPException, Depends

from ..models import (
    QueryRequest, QueryResponse,
    ChatRequest, ChatResponse,
    SourceInfo, StatsResponse
)
from ...rag import RAGChain

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


def get_rag_chain():
    """Dependency to get RAG chain instance."""
    from ..main import rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized. Check your API key.")
    return rag_chain


@router.post("", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    rag: RAGChain = Depends(get_rag_chain)
):
    """
    Query the knowledge base with a question.
    
    Returns an AI-generated answer grounded in the ingested documents,
    along with source citations.
    """
    try:
        # Update retriever top_k if specified
        rag.retriever.top_k = request.top_k
        
        # Process query through RAG pipeline
        result = rag.query(request.question)
        
        # Format sources
        sources = [
            SourceInfo(
                source=s.get("source", "Unknown"),
                title=s.get("title", ""),
                relevance_score=s.get("relevance_score", 0)
            )
            for s in result.get("sources", [])
        ]
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            context_used=result.get("context_used", True),
            documents_retrieved=result.get("documents_retrieved", 0)
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag: RAGChain = Depends(get_rag_chain)
):
    """
    Chat with the knowledge base (supports conversation history).
    
    Maintains context from previous messages in the conversation.
    """
    try:
        # Convert to dict format
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        # Process through RAG chat
        result = rag.chat(messages)
        
        # Format sources
        sources = [
            SourceInfo(
                source=s.get("source", "Unknown"),
                title=s.get("title", ""),
                relevance_score=s.get("relevance_score", 0)
            )
            for s in result.get("sources", [])
        ]
        
        return ChatResponse(
            answer=result["answer"],
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats(rag: RAGChain = Depends(get_rag_chain)):
    """
    Get statistics about the knowledge base.
    """
    try:
        stats = rag.get_stats()
        return StatsResponse(
            collection_name=stats.get("collection_name", ""),
            document_count=stats.get("document_count", 0),
            embedding_model=stats.get("embedding_model", ""),
            llm_model=stats.get("llm_model", "")
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===================== KNOWLEDGE GRAPH ENDPOINTS =====================

from ...knowledge_graph import KnowledgeGraph
from ...knowledge_graph.graph import get_knowledge_graph


@router.get("/knowledge-graph")
async def get_knowledge_graph_data(max_nodes: int = 200):
    """
    Get knowledge graph data for D3.js visualization.
    
    Returns nodes and links in D3 force-directed graph format.
    """
    try:
        kg = get_knowledge_graph()
        return kg.to_d3_format(max_nodes=max_nodes)
    except Exception as e:
        logger.error(f"Error getting knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/stats")
async def get_knowledge_graph_stats():
    """
    Get statistics about the knowledge graph.
    """
    try:
        kg = get_knowledge_graph()
        return kg.get_stats()
    except Exception as e:
        logger.error(f"Error getting knowledge graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/entity/{entity_name}")
async def get_entity_details(entity_name: str):
    """
    Get details about a specific entity and its relationships.
    """
    try:
        kg = get_knowledge_graph()
        entity = kg.find_entity(entity_name)
        
        if not entity:
            raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found")
        
        related = kg.get_related(entity["id"])
        
        return {
            "entity": entity,
            "related": related
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/search")
async def search_knowledge_graph(q: str, entity_type: str = None):
    """
    Search for entities in the knowledge graph.
    """
    try:
        kg = get_knowledge_graph()
        results = kg.search_entities(q, entity_type)
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Error searching knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/brands")
async def get_brands():
    """
    Get all brands in the knowledge graph.
    """
    try:
        kg = get_knowledge_graph()
        brands = kg.get_entities_by_type("brand")
        return {"brands": brands, "count": len(brands)}
    except Exception as e:
        logger.error(f"Error getting brands: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/categories")
async def get_categories():
    """
    Get all categories in the knowledge graph.
    """
    try:
        kg = get_knowledge_graph()
        categories = kg.get_entities_by_type("category")
        return {"categories": categories, "count": len(categories)}
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/brand/{brand_name}/products")
async def get_products_by_brand(brand_name: str):
    """
    Get all products for a specific brand.
    """
    try:
        kg = get_knowledge_graph()
        products = kg.get_products_by_brand(brand_name)
        return {"brand": brand_name, "products": products, "count": len(products)}
    except Exception as e:
        logger.error(f"Error getting products by brand: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/category/{category_name}/products")
async def get_products_by_category(category_name: str):
    """
    Get all products in a specific category.
    """
    try:
        kg = get_knowledge_graph()
        products = kg.get_products_by_category(category_name)
        return {"category": category_name, "products": products, "count": len(products)}
    except Exception as e:
        logger.error(f"Error getting products by category: {e}")
        raise HTTPException(status_code=500, detail=str(e))

