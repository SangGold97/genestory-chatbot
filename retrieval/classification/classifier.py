import numpy as np
from typing import Dict, List, Tuple
from sklearn.neighbors import NearestNeighbors
from loguru import logger

from models.embedding.manager import embedding_manager
from retrieval.classification.sample_queries import SAMPLE_QUERIES
from config.settings import settings


class QueryClassifier:
    """
    Classifies user queries into two categories:
    - Label 1: Greeting/general questions (not related to company)
    - Label 2: Questions about Genestory services/products/information
    
    Uses sentence-transformers embeddings and k-NN with cosine similarity.
    """
    
    def __init__(self):
        self.embedding_manager = embedding_manager
        self.threshold = settings.CLASSIFICATION_THRESHOLD
        self.sample_queries = SAMPLE_QUERIES
        self.is_trained = False
        
        # Training data
        self.training_queries = []
        self.training_labels = []
        self.training_embeddings = None
        
        # k-NN models for each label
        self.knn_models = {}
        self.label_embeddings = {}
        
    async def prepare_training_data(self):
        """Prepare training data from sample queries."""
        try:
            logger.info("Preparing training data for query classification")
            
            # Ensure embedding model is loaded
            if not self.embedding_manager.is_loaded():
                await self.embedding_manager.load_model()
            
            # Prepare training data
            self.training_queries = []
            self.training_labels = []
            
            # Add label 1 queries (general/greeting)
            for query in self.sample_queries["label_1"]:
                self.training_queries.append(query)
                self.training_labels.append(1)
            
            # Add label 2 queries (company-related)
            for query in self.sample_queries["label_2"]:
                self.training_queries.append(query)
                self.training_labels.append(2)
            
            # Generate embeddings for training queries
            self.training_embeddings = self.embedding_manager.encode(self.training_queries)
            
            logger.info(f"Prepared {len(self.training_queries)} training queries")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return False
    
    async def train_classifier(self, k_neighbors: int = 5):
        """Train the k-NN classifier."""
        try:
            if not self.training_embeddings is not None:
                await self.prepare_training_data()
            
            logger.info("Training query classifier")
            
            # Separate embeddings by label
            label_1_embeddings = []
            label_2_embeddings = []
            
            for i, label in enumerate(self.training_labels):
                if label == 1:
                    label_1_embeddings.append(self.training_embeddings[i])
                else:
                    label_2_embeddings.append(self.training_embeddings[i])
            
            # Convert to numpy arrays
            self.label_embeddings[1] = np.array(label_1_embeddings)
            self.label_embeddings[2] = np.array(label_2_embeddings)
            
            # Train k-NN models for each label
            for label in [1, 2]:
                knn = NearestNeighbors(
                    n_neighbors=min(k_neighbors, len(self.label_embeddings[label])),
                    metric='cosine'
                )
                knn.fit(self.label_embeddings[label])
                self.knn_models[label] = knn
            
            self.is_trained = True
            logger.info("Query classifier trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training classifier: {str(e)}")
            return False
    
    async def classify_query(self, query: str) -> Tuple[int, float]:
        """
        Classify a query into label 1 or 2.
        
        Args:
            query: User query to classify
            
        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        try:
            if not self.is_trained:
                await self.train_classifier()
            
            # Encode the query
            query_embedding = self.embedding_manager.encode_query(query)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Get distances to nearest neighbors for each label
            scores = {}
            
            for label in [1, 2]:
                distances, indices = self.knn_models[label].kneighbors(query_embedding)
                
                # Convert cosine distance to similarity
                # Cosine distance = 1 - cosine similarity
                similarities = 1 - distances[0]
                
                # Use the maximum similarity as the score for this label
                scores[label] = np.max(similarities)
            
            # Determine the predicted label
            predicted_label = max(scores, key=scores.get)
            confidence = float(scores[predicted_label])  # Convert numpy type to Python float
            
            logger.debug(f"Query: '{query}' | Label: {predicted_label} | Confidence: {confidence:.3f}")
            
            return predicted_label, confidence
            
        except Exception as e:
            logger.error(f"Error classifying query: {str(e)}")
            return 2, 0.0  # Default to company-related if error
    
    async def is_company_related(self, query: str) -> bool:
        """
        Check if a query is related to the company (label 2).
        
        Args:
            query: User query
            
        Returns:
            True if company-related, False otherwise
        """
        label, confidence = await self.classify_query(query)
        
        # Return True if label is 2 (company-related) and confidence is above threshold
        return label == 2 and confidence >= self.threshold
    
    async def get_classification_details(self, query: str) -> Dict:
        """
        Get detailed classification information for a query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with classification details
        """
        label, confidence = await self.classify_query(query)
        
        return {
            "query": query,
            "predicted_label": int(label),  # Ensure Python int
            "label_description": "Company-related" if label == 2 else "General/Greeting",
            "confidence": float(confidence),  # Ensure Python float
            "is_company_related": label == 2 and confidence >= self.threshold,
            "threshold": float(self.threshold)  # Ensure Python float
        }
    
    def get_model_info(self) -> Dict:
        """Get classifier model information."""
        return {
            "is_trained": self.is_trained,
            "threshold": self.threshold,
            "num_training_queries": len(self.training_queries) if self.training_queries else 0,
            "num_label_1": len(self.sample_queries["label_1"]),
            "num_label_2": len(self.sample_queries["label_2"])
        }


# Global instance
query_classifier = QueryClassifier()
