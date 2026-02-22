"""
Centralized configuration management using Pydantic Settings.
All settings can be overridden via environment variables.
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = Field(default="GenAI RAG Q&A System")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    
    # API Server
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    
    # Google Gemini (FREE LLM)
    google_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-pro")
    
    # Embeddings (FREE - Local HuggingFace)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    
    # Vector Database (FREE - ChromaDB Local)
    chroma_persist_dir: str = Field(default="./data/chroma_db")
    chroma_collection_name: str = Field(default="documents")
    
    # Data Storage
    upload_dir: str = Field(default="./data/uploads")
    max_file_size_mb: int = Field(default=10)
    
    # RAG Settings
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    top_k_results: int = Field(default=5)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def upload_path(self) -> Path:
        """Get upload directory as Path object."""
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def chroma_path(self) -> Path:
        """Get ChromaDB directory as Path object."""
        path = Path(self.chroma_persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global settings instance
settings = Settings()
