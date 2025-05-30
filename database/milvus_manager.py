import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import (
    connections, 
    Collection, 
    CollectionSchema, 
    FieldSchema, 
    DataType,
    utility
)
import numpy as np
from loguru import logger
from config.settings import settings


class MilvusManager:
    """Manages Milvus vector database operations."""
    
    def __init__(self):
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.dimension = settings.MILVUS_DIMENSION
        self.collection = None
        self.is_connected = False
        
    async def connect(self) -> bool:
        """Connect to Milvus server."""
        try:
            connections.connect("default", host=self.host, port=self.port)
            self.is_connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Milvus server."""
        try:
            connections.disconnect("default")
            self.is_connected = False
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {str(e)}")
    
    async def create_collection(self) -> bool:
        """Create the collection if it doesn't exist."""
        try:
            if not self.is_connected:
                await self.connect()
            
            # Check if collection already exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' already exists")
                return True
            
            # Define collection schema
            fields = [
                FieldSchema(
                    name="id", 
                    dtype=DataType.INT64, 
                    is_primary=True, 
                    auto_id=True
                ),
                FieldSchema(
                    name="document_id", 
                    dtype=DataType.VARCHAR, 
                    max_length=512
                ),
                FieldSchema(
                    name="content", 
                    dtype=DataType.VARCHAR, 
                    max_length=65535
                ),
                FieldSchema(
                    name="source", 
                    dtype=DataType.VARCHAR, 
                    max_length=512
                ),
                FieldSchema(
                    name="metadata", 
                    dtype=DataType.VARCHAR, 
                    max_length=2048
                ),
                FieldSchema(
                    name="embedding", 
                    dtype=DataType.FLOAT_VECTOR, 
                    dim=self.dimension
                )
            ]
            
            schema = CollectionSchema(
                fields=fields, 
                description=f"Genestory knowledge base collection"
            )
            
            # Create collection
            self.collection = Collection(
                name=self.collection_name, 
                schema=schema
            )
            
            # Create index for vector field
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            self.collection.create_index(
                field_name="embedding", 
                index_params=index_params
            )
            
            logger.info(f"Collection '{self.collection_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            return False
    
    async def insert_documents(
        self, 
        documents: List[str], 
        embeddings: np.ndarray,
        document_ids: List[str],
        sources: List[str],
        metadata: List[str] = None
    ) -> bool:
        """
        Insert documents with their embeddings into the collection.
        
        Args:
            documents: List of document texts
            embeddings: Document embeddings
            document_ids: Document IDs
            sources: Document sources
            metadata: Document metadata (optional)
        """
        try:
            if not self.collection:
                await self.create_collection()
            
            if metadata is None:
                metadata = ["{}"] * len(documents)
            
            # Prepare data for insertion
            data = [
                document_ids,
                documents,
                sources,
                metadata,
                embeddings.tolist()
            ]
            
            # Insert data
            insert_result = self.collection.insert(data)
            
            # Flush to ensure data is written
            self.collection.flush()
            
            logger.info(f"Inserted {len(documents)} documents into collection")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting documents: {str(e)}")
            return False
    
    async def search_similar(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of similar documents with metadata
        """
        try:
            if not self.collection:
                await self.create_collection()
            
            # Load collection if not loaded
            self.collection.load()
            
            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["document_id", "content", "source", "metadata"]
            )
            
            # Process results
            similar_docs = []
            for result in results[0]:
                score = float(result.score)
                
                # Apply score threshold if specified
                if score_threshold and score < score_threshold:
                    continue
                
                # Parse metadata JSON string back to dictionary
                metadata_str = result.entity.get("metadata")
                try:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                except (json.JSONDecodeError, TypeError):
                    # Handle invalid JSON or None values
                    metadata = {}
                    logger.warning(f"Failed to parse metadata JSON: {metadata_str}")
                
                doc = {
                    "id": result.entity.get("document_id"),
                    "content": result.entity.get("content"),
                    "source": result.entity.get("source"),
                    "metadata": metadata,
                    "score": score
                }
                similar_docs.append(doc)
            
            logger.info(f"Found {len(similar_docs)} similar documents")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            return []
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by their IDs."""
        try:
            if not self.collection:
                return False
            
            # Delete documents
            expr = f"document_id in {document_ids}"
            self.collection.delete(expr)
            
            logger.info(f"Deleted documents with IDs: {document_ids}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    async def clear_collection(self) -> bool:
        """Clear all data from the collection."""
        try:
            if not self.collection:
                return False
            
            # Delete all entities
            self.collection.delete(expr="id >= 0")
            
            logger.info("Cleared all data from collection")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            if not self.collection:
                return {}
            
            stats = self.collection.num_entities
            
            return {
                "collection_name": self.collection_name,
                "num_entities": stats,
                "dimension": self.dimension,
                "is_connected": self.is_connected
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}


# Global instance
milvus_manager = MilvusManager()
