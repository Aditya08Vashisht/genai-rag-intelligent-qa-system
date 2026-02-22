"""
API Models - Pydantic models for request/response validation.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl


# ==================== Ingestion Models ====================

class URLIngestRequest(BaseModel):
    """Request to ingest content from URLs."""
    urls: List[str] = Field(..., description="List of URLs to scrape")
    
    class Config:
        json_schema_extra = {
            "example": {
                "urls": ["https://example.com/article1", "https://example.com/article2"]
            }
        }


class TextIngestRequest(BaseModel):
    """Request to ingest raw text."""
    text: str = Field(..., description="Text content to ingest")
    source: str = Field(default="user_input", description="Source identifier")
    title: str = Field(default="", description="Optional title")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is the content to add to the knowledge base...",
                "source": "manual_entry",
                "title": "My Document"
            }
        }


class IngestResponse(BaseModel):
    """Response after ingestion."""
    success: bool
    message: str
    documents_added: int = 0
    chunks_created: int = 0


# ==================== Query Models ====================

class QueryRequest(BaseModel):
    """Request to query the knowledge base."""
    question: str = Field(..., description="Question to ask")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is this document about?",
                "top_k": 5
            }
        }


class SourceInfo(BaseModel):
    """Information about a source document."""
    source: str
    title: str = ""
    relevance_score: float


class QueryResponse(BaseModel):
    """Response to a query."""
    answer: str
    sources: List[SourceInfo] = []
    context_used: bool = True
    documents_retrieved: int = 0


# ==================== Chat Models ====================

class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat endpoint."""
    messages: List[ChatMessage]
    
    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "What is RAG?"},
                    {"role": "assistant", "content": "RAG stands for..."},
                    {"role": "user", "content": "How does it work?"}
                ]
            }
        }


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    answer: str
    sources: List[SourceInfo] = []


# ==================== System Models ====================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"


class StatsResponse(BaseModel):
    """System statistics response."""
    collection_name: str
    document_count: int
    embedding_model: str
    llm_model: str
