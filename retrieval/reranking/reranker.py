from typing import List, Dict, Any, Tuple
import numpy as np
from loguru import logger

from models.embedding.manager import embedding_manager
from config.settings import settings


class DocumentReranker:
    """
    Reranks retrieved documents to improve relevance ordering.
    Uses cross-encoder or advanced similarity scoring.
    """
    
    def __init__(self):
        self.embedding_manager = embedding_manager
        self.rerank_top_k = settings.RERANK_TOP_K
        
    async def rerank_documents(
        self, 
        query: str, 
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to the query.
        
        Args:
            query: User query
            documents: List of retrieved documents
            top_k: Number of top documents to return after reranking
            
        Returns:
            Reranked list of documents
        """
        try:
            if not documents:
                return documents
            
            top_k = top_k or self.rerank_top_k
            top_k = min(top_k, len(documents))
            
            logger.info(f"Reranking {len(documents)} documents")
            
            # Ensure embedding model is loaded
            if not self.embedding_manager.is_loaded():
                await self.embedding_manager.load_model()
            
            # Encode query
            query_embedding = self.embedding_manager.encode_query(query)
            
            # Extract document contents and encode them
            doc_contents = [doc.get("content", "") for doc in documents]
            doc_embeddings = self.embedding_manager.encode_documents(doc_contents)
            
            # Compute detailed similarities
            reranked_docs = []
            for i, doc in enumerate(documents):
                # Compute similarity between query and document
                similarity = self.embedding_manager.compute_similarity(
                    query_embedding, 
                    doc_embeddings[i]
                )
                
                # Create enhanced document with rerank score
                enhanced_doc = doc.copy()
                enhanced_doc["rerank_score"] = float(similarity)
                enhanced_doc["original_rank"] = i + 1
                
                # Combine original score with rerank score
                # You can adjust this weighting as needed
                original_score = doc.get("score", 0.0)
                combined_score = (0.3 * original_score) + (0.7 * similarity)
                enhanced_doc["combined_score"] = float(combined_score)  # Ensure Python float
                
                reranked_docs.append(enhanced_doc)
            
            # Sort by combined score (descending)
            reranked_docs.sort(key=lambda x: x["combined_score"], reverse=True)
            
            # Return top-k documents
            final_docs = reranked_docs[:top_k]
            
            logger.info(f"Reranked and returned top {len(final_docs)} documents")
            return final_docs
            
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            return documents  # Return original documents if reranking fails
    
    async def rerank_with_context_awareness(
        self, 
        query: str, 
        documents: List[Dict[str, Any]],
        conversation_history: List[str] = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents with conversation context awareness.
        
        Args:
            query: Current user query
            documents: List of retrieved documents
            conversation_history: Previous conversation turns
            top_k: Number of top documents to return
            
        Returns:
            Context-aware reranked documents
        """
        try:
            if not documents:
                return documents
            
            # For now, use the basic reranking
            # In a more advanced implementation, you could:
            # 1. Combine query with conversation history
            # 2. Weight documents based on previous interactions
            # 3. Consider topic coherence across conversation
            
            if conversation_history:
                # Create enhanced query with context
                context = " ".join(conversation_history[-3:])  # Last 3 turns
                enhanced_query = f"{context} {query}"
                return await self.rerank_documents(enhanced_query, documents, top_k)
            else:
                return await self.rerank_documents(query, documents, top_k)
            
        except Exception as e:
            logger.error(f"Error in context-aware reranking: {str(e)}")
            return await self.rerank_documents(query, documents, top_k)
    
    def calculate_diversity_score(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate diversity scores to avoid redundant documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Documents with diversity scores
        """
        try:
            if len(documents) <= 1:
                return documents
            
            # Extract document contents
            doc_contents = [doc.get("content", "") for doc in documents]
            doc_embeddings = self.embedding_manager.encode_documents(doc_contents)
            
            # Calculate pairwise similarities
            for i, doc in enumerate(documents):
                similarities = []
                for j, other_embedding in enumerate(doc_embeddings):
                    if i != j:
                        similarity = self.embedding_manager.compute_similarity(
                            doc_embeddings[i], other_embedding
                        )
                        similarities.append(similarity)
                
                # Diversity score is inverse of average similarity
                avg_similarity = np.mean(similarities) if similarities else 0
                diversity_score = 1.0 - float(avg_similarity)  # Ensure Python float
                doc["diversity_score"] = diversity_score
            
            return documents
            
        except Exception as e:
            logger.error(f"Error calculating diversity scores: {str(e)}")
            return documents
    
    async def rerank_with_diversity(
        self, 
        query: str, 
        documents: List[Dict[str, Any]],
        diversity_weight: float = 0.2,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents considering both relevance and diversity.
        
        Args:
            query: User query
            documents: List of retrieved documents
            diversity_weight: Weight for diversity in scoring (0-1)
            top_k: Number of top documents to return
            
        Returns:
            Reranked documents with diversity consideration
        """
        try:
            # First, do regular reranking
            reranked_docs = await self.rerank_documents(query, documents, len(documents))
            
            # Calculate diversity scores
            docs_with_diversity = self.calculate_diversity_score(reranked_docs)
            
            # Combine relevance and diversity scores
            for doc in docs_with_diversity:
                relevance_score = doc.get("combined_score", doc.get("score", 0.0))
                diversity_score = doc.get("diversity_score", 0.0)
                
                final_score = (
                    (1 - diversity_weight) * relevance_score + 
                    diversity_weight * diversity_score
                )
                doc["final_score"] = float(final_score)  # Ensure Python float
            
            # Sort by final score
            docs_with_diversity.sort(key=lambda x: x["final_score"], reverse=True)
            
            # Return top-k
            top_k = top_k or self.rerank_top_k
            return docs_with_diversity[:top_k]
            
        except Exception as e:
            logger.error(f"Error in diversity-aware reranking: {str(e)}")
            return await self.rerank_documents(query, documents, top_k)
    
    def get_reranker_info(self) -> Dict[str, Any]:
        """Get reranker information."""
        return {
            "rerank_top_k": self.rerank_top_k,
            "embedding_model_loaded": self.embedding_manager.is_loaded()
        }


# Global instance
document_reranker = DocumentReranker()
