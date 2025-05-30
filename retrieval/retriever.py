from typing import List, Dict, Any, Tuple
import numpy as np
from loguru import logger

from database.milvus_manager import milvus_manager
from models.embedding.manager import embedding_manager
from config.settings import settings


class DocumentRetriever:
    """Handles document retrieval from the vector database."""
    
    def __init__(self):
        self.milvus_manager = milvus_manager
        self.embedding_manager = embedding_manager
        self.top_k = settings.RETRIEVAL_TOP_K
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        
    async def retrieve_documents(
        self, 
        query: str, 
        top_k: int = None,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of retrieved documents with metadata
        """
        try:
            top_k = top_k or self.top_k
            score_threshold = score_threshold or self.similarity_threshold
            
            logger.info(f"Retrieving documents for query: '{query}'")
            
            # Ensure embedding model is loaded
            if not self.embedding_manager.is_loaded():
                await self.embedding_manager.load_model()
            
            # Encode the query
            query_embedding = self.embedding_manager.encode_query(query)
            
            # Search for similar documents
            similar_docs = await self.milvus_manager.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            logger.info(f"Retrieved {len(similar_docs)} documents")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    async def retrieve_with_metadata(
        self, 
        query: str, 
        filters: Dict[str, Any] = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with additional filtering based on metadata.
        
        Args:
            query: User query
            filters: Additional filters to apply
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # For now, this is the same as retrieve_documents
        # In a more advanced implementation, you could add metadata filtering
        return await self.retrieve_documents(query, top_k)
    
    def format_retrieved_documents(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string for the LLM.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "").strip()
            source = doc.get("source", "Unknown")
            score = doc.get("score", 0.0)
            
            context_part = f"""Document {i} (Source: {source}, Relevance: {score:.3f}):
{content}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        stats = await self.milvus_manager.get_collection_stats()
        stats.update({
            "retrieval_top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "embedding_model_loaded": self.embedding_manager.is_loaded()
        })
        return stats


# Global instance
document_retriever = DocumentRetriever()
