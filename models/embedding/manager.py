import os
import numpy as np
from typing import List, Union, Dict, Any
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from loguru import logger
from config.settings import settings


class EmbeddingModelManager:
    """Manages the local embedding model for text embeddings."""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.model_path = settings.EMBEDDING_MODEL_PATH
        self.model = None
        self.dimension = settings.MILVUS_DIMENSION
        
    async def download_model(self) -> bool:
        """Download the embedding model from Hugging Face if not already present."""
        try:
            logger.info(f"Downloading embedding model {self.model_name}")
            
            # Create model directory if it doesn't exist
            os.makedirs(self.model_path, exist_ok=True)
            
            # Download model
            snapshot_download(
                repo_id=self.model_name,
                local_dir=self.model_path,
                resume_download=True,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"Embedding model {self.model_name} downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading embedding model: {str(e)}")
            return False
    
    async def load_model(self) -> bool:
        """Load the embedding model."""
        try:
            logger.info(f"Loading embedding model from {self.model_path}")
            
            # Check if model exists locally, download if not
            if not os.path.exists(self.model_path) or not os.listdir(self.model_path):
                logger.info("Embedding model not found locally, downloading...")
                if not await self.download_model():
                    return False
            
            # Load the model
            self.model = SentenceTransformer(self.model_path)
            
            # Verify model dimension
            test_embedding = self.model.encode(["test"])
            actual_dimension = len(test_embedding[0])
            
            if actual_dimension != self.dimension:
                logger.warning(
                    f"Model dimension {actual_dimension} doesn't match expected {self.dimension}. "
                    f"Updating dimension setting."
                )
                self.dimension = actual_dimension
            
            logger.info(f"Embedding model loaded successfully with dimension {self.dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            return False
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            normalize: Whether to normalize embeddings
            
        Returns:
            numpy array of embeddings
        """
        if not self.model:
            raise ValueError("Embedding model not loaded. Call load_model() first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into embedding."""
        return self.encode(query)[0]
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode multiple documents into embeddings."""
        return self.encode(documents)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def compute_similarities(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between a query and multiple documents.
        
        Args:
            query_embedding: Query embedding
            doc_embeddings: Document embeddings
            
        Returns:
            Array of similarity scores
        """
        # Normalize embeddings
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarities
        similarities = np.dot(doc_embeddings, query_embedding)
        return similarities
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        doc_embeddings: np.ndarray, 
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar documents to a query.
        
        Args:
            query_embedding: Query embedding
            doc_embeddings: Document embeddings
            top_k: Number of top similar documents to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = self.compute_similarities(query_embedding, doc_embeddings)
        
        # Get top-k indices and scores
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return [(int(idx), float(score)) for idx, score in zip(top_indices, top_scores)]
    
    def is_loaded(self) -> bool:
        """Check if embedding model is loaded."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get embedding model information."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "dimension": self.dimension,
            "is_loaded": self.is_loaded()
        }


# Global instance
embedding_manager = EmbeddingModelManager()
