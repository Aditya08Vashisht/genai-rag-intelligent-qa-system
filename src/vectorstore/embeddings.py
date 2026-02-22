"""
Embedding Model - Generate embeddings using HuggingFace (FREE)

Uses sentence-transformers for local embedding generation.
No API keys required - runs entirely on your machine.
"""
import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Embedding model using HuggingFace sentence-transformers.
    
    100% FREE - runs locally, no API keys needed.
    
    Popular models:
    - all-MiniLM-L6-v2: Fast, good quality (default)
    - all-mpnet-base-v2: Better quality, slower
    - paraphrase-MiniLM-L6-v2: Good for paraphrase detection
    """
    
    # Model dimensions for reference
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model name (downloads automatically)
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        logger.info(f"Initializing embedding model: {model_name}")
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading model {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        return self._model
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        return self.MODEL_DIMENSIONS.get(self.model_name, 384)
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        if not text.strip():
            return [0.0] * self.embedding_dimension
        
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=False
        )
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return [emb.tolist() for emb in embeddings]
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Note: Some models use different embeddings for queries vs documents.
        This method can be extended to support asymmetric search.
        
        Args:
            query: Search query
            
        Returns:
            Query embedding vector
        """
        return self.embed_text(query)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0 to 1)
        """
        from sentence_transformers import util
        
        emb1 = self.model.encode(text1, convert_to_tensor=True)
        emb2 = self.model.encode(text2, convert_to_tensor=True)
        
        similarity = util.cos_sim(emb1, emb2)
        return float(similarity[0][0])
