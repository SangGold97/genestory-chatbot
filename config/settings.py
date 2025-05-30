import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Application
    APP_NAME: str = "Genestory Chatbot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Settings
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Streamlit Settings
    STREAMLIT_HOST: str = "localhost"
    STREAMLIT_PORT: int = 8501
    
    # Model Settings
    LLM_MODEL_NAME: str = "google/gemma-2-2b-it"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Model Paths
    MODELS_DIR: str = "models"
    LLM_MODEL_PATH: str = "models/llm/gemma-2-2b-it"
    EMBEDDING_MODEL_PATH: str = "models/embedding/all-MiniLM-L6-v2"
    
    # Milvus Settings
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_NAME: str = "genestory_knowledge"
    MILVUS_DIMENSION: int = 384  # all-MiniLM-L6-v2 embedding dimension
    
    # Retrieval Settings
    RETRIEVAL_TOP_K: int = 20
    RERANK_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.5
    
    # Knowledge Base
    KNOWLEDGE_BASE_DIR: str = "knowledge_base"
    DOCUMENTS_DIR: str = "knowledge_base/documents"
    SAMPLE_DATA_DIR: str = "knowledge_base/sample_data"
    
    # Query Classification
    CLASSIFICATION_THRESHOLD: float = 0.1
    
    # Generation Settings
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/chatbot.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
