"""
Retriever - Semantic search for relevant documents.

Retrieves the most relevant document chunks for a given query
using vector similarity search.
"""
import logging
from typing import List, Dict, Any, Optional
from ..vectorstore import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Document retriever using semantic similarity search.
    
    Features:
    - Semantic search via embeddings
    - Metadata filtering
    - Score thresholding
    - Configurable top-k
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.3
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store instance
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score (0-1)
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of results (overrides default top_k)
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant documents with scores
        """
        num_results = k or self.top_k
        
        # Search vector store
        results = self.vector_store.search(
            query=query,
            k=num_results,
            filter_metadata=filter_metadata
        )
        
        # Filter by score threshold
        filtered_results = [
            r for r in results
            if r.get("score", 0) >= self.score_threshold
        ]
        
        logger.info(
            f"Retrieved {len(filtered_results)}/{len(results)} documents "
            f"above threshold {self.score_threshold}"
        )
        
        return filtered_results
    
    def retrieve_with_context(
        self,
        query: str,
        k: Optional[int] = None
    ) -> str:
        """
        Retrieve documents and format as context string.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            Formatted context string for LLM
        """
        results = self.retrieve(query, k=k)
        
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.get("metadata", {}).get("source", "Unknown")
            content = doc.get("content", "")
            score = doc.get("score", 0)
            
            context_parts.append(
                f"[Source {i}: {source} (relevance: {score:.2f})]\n{content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source citations from results.
        
        Args:
            results: List of retrieved documents
            
        Returns:
            List of source citations
        """
        sources = []
        seen = set()
        
        for doc in results:
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")
            
            if source not in seen:
                seen.add(source)
                sources.append({
                    "source": source,
                    "title": metadata.get("document_title", ""),
                    "relevance_score": doc.get("score", 0)
                })
        
        return sources
