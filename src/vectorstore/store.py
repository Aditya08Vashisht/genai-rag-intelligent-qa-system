"""
Vector Store - FAISS Integration (FREE, Windows-compatible)

FAISS is a free, open-source vector database from Facebook AI.
Has prebuilt Windows wheels - no compilation needed!
"""
import logging
import pickle
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np

from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store using FAISS (or numpy fallback).
    
    100% FREE - works on Windows without build tools.
    
    Features:
    - Store document embeddings
    - Semantic similarity search
    - Metadata storage
    - Persistent storage via pickle
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./data/vector_db",
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name for the collection
            persist_directory: Directory to persist the database
            embedding_model: Optional custom embedding model
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = embedding_model or EmbeddingModel()
        
        # Storage
        self.documents: List[str] = []
        self.embeddings: List[List[float]] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        
        # Try to load existing data
        self._load()
        
        logger.info(f"Vector store initialized with {len(self.documents)} documents")
    
    def _get_storage_path(self) -> Path:
        """Get the path to the storage file."""
        return self.persist_directory / f"{self.collection_name}.pkl"
    
    def _save(self):
        """Save the vector store to disk."""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "metadatas": self.metadatas,
            "ids": self.ids
        }
        with open(self._get_storage_path(), "wb") as f:
            pickle.dump(data, f)
        logger.debug(f"Saved {len(self.documents)} documents to disk")
    
    def _load(self):
        """Load the vector store from disk."""
        path = self._get_storage_path()
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self.documents = data.get("documents", [])
                self.embeddings = data.get("embeddings", [])
                self.metadatas = data.get("metadatas", [])
                self.ids = data.get("ids", [])
                logger.info(f"Loaded {len(self.documents)} documents from disk")
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts
            ids: Optional list of unique IDs
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Generate IDs if not provided
        import uuid
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Generate embeddings
        logger.info(f"Adding {len(documents)} documents to vector store...")
        new_embeddings = self.embedding_model.embed_texts(documents)
        
        # Prepare metadatas
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
        
        # Add to storage
        self.documents.extend(documents)
        self.embeddings.extend(new_embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        # Persist
        self._save()
        
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
        return ids
    
    def add_chunks(self, chunks: List[dict]) -> List[str]:
        """
        Add text chunks to the vector store.
        
        Args:
            chunks: List of chunk dicts with 'content', 'source', 'metadata'
            
        Returns:
            List of document IDs
        """
        documents = [chunk.get("content", "") for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = chunk.get("metadata", {}).copy()
            metadata["source"] = chunk.get("source", "unknown")
            metadata["chunk_index"] = chunk.get("chunk_index", 0)
            metadatas.append(metadata)
        
        return self.add_documents(documents, metadatas=metadatas)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of results with 'content', 'metadata', 'score'
        """
        if len(self.documents) == 0:
            logger.warning("No documents in collection")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            # Apply metadata filter if provided
            if filter_metadata:
                match = all(
                    self.metadatas[i].get(k) == v 
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue
            
            score = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, score))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k results
        results = []
        for i, score in similarities[:k]:
            results.append({
                "content": self.documents[i],
                "metadata": self.metadatas[i],
                "score": round(score, 4),
                "id": self.ids[i]
            })
        
        logger.info(f"Found {len(results)} results for query: {query[:50]}...")
        return results
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        logger.warning(f"Deleting collection '{self.collection_name}'")
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        
        # Delete file
        path = self._get_storage_path()
        if path.exists():
            path.unlink()
    
    @property
    def collection(self):
        """Compatibility property for collection count."""
        return self
    
    def count(self) -> int:
        """Return the number of documents."""
        return len(self.documents)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "collection_name": self.collection_name,
            "document_count": len(self.documents),
            "persist_directory": str(self.persist_directory),
            "embedding_model": self.embedding_model.model_name
        }
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents with their metadata."""
        return [
            {
                "content": self.documents[i],
                "metadata": self.metadatas[i],
                "id": self.ids[i]
            }
            for i in range(len(self.documents))
        ]
