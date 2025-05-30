import time
import asyncio
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
from loguru import logger
import json

from backend.models import (
    QueryRequest, QueryResponse, 
    DocumentUploadRequest, DocumentUploadResponse,
    ModelLoadRequest, ModelLoadResponse,
    HealthResponse, SystemInfoResponse, ErrorResponse
)
from workflow.workflow import chatbot_workflow
from models.llm.manager import llm_manager
from models.embedding.manager import embedding_manager
from database.milvus_manager import milvus_manager
from retrieval.classification.classifier import query_classifier
from retrieval.retriever import document_retriever
from config.settings import settings

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Genestory Local Chatbot API with RAG and LangGraph"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store app start time for uptime calculation
app_start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Genestory Chatbot API...")
    
    # Initialize database connection
    try:
        await milvus_manager.connect()
        await milvus_manager.create_collection()
        logger.info("Milvus database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Genestory Chatbot API...")
    
    try:
        await milvus_manager.disconnect()
        logger.info("Disconnected from database")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.APP_NAME} API",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check model statuses
        models_status = {
            "llm_loaded": llm_manager.is_loaded(),
            "embedding_loaded": embedding_manager.is_loaded(),
            "classifier_trained": query_classifier.is_trained
        }
        
        # Check database status
        db_stats = await milvus_manager.get_collection_stats()
        
        uptime = time.time() - app_start_time
        
        return HealthResponse(
            status="healthy",
            models=models_status,
            database=db_stats,
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """Process a chat query."""
    try:
        start_time = time.time()
        
        # Process the query through the workflow
        result = await chatbot_workflow.process_query(
            query=request.query,
            conversation_history=request.conversation_history
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return QueryResponse(
            query=result["query"],
            response=result["response"],
            classification=result["classification"],
            num_documents_retrieved=result["num_documents_retrieved"],
            num_documents_reranked=result["num_documents_reranked"],
            metadata=result["metadata"],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: QueryRequest):
    """Stream chat response."""
    async def generate_response():
        try:
            # For now, return the regular response
            # In a full streaming implementation, you'd yield tokens as they're generated
            result = await chatbot_workflow.process_query(
                query=request.query,
                conversation_history=request.conversation_history
            )
            
            # Stream the response
            response_data = {
                "type": "response",
                "content": result["response"],
                "metadata": result["metadata"]
            }
            
            yield f"data: {json.dumps(response_data)}\n\n"
            
        except Exception as e:
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.post("/models/load", response_model=ModelLoadResponse)
async def load_model(request: ModelLoadRequest):
    """Load a specific model."""
    try:
        if request.model_type.lower() == "llm":
            success = await llm_manager.load_model()
            model_info = llm_manager.get_model_info()
            message = "LLM model loaded successfully" if success else "Failed to load LLM model"
            
        elif request.model_type.lower() == "embedding":
            success = await embedding_manager.load_model()
            model_info = embedding_manager.get_model_info()
            message = "Embedding model loaded successfully" if success else "Failed to load embedding model"
            
        elif request.model_type.lower() == "classifier":
            success = await query_classifier.train_classifier()
            model_info = query_classifier.get_model_info()
            message = "Query classifier trained successfully" if success else "Failed to train query classifier"
            
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        return ModelLoadResponse(
            success=success,
            message=message,
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(request: DocumentUploadRequest):
    """Upload a document to the knowledge base."""
    try:
        # Ensure embedding model is loaded
        if not embedding_manager.is_loaded():
            await embedding_manager.load_model()
        
        # Generate embedding for the document
        embedding = embedding_manager.encode_query(request.content)
        
        # Insert into database
        success = await milvus_manager.insert_documents(
            documents=[request.content],
            embeddings=embedding.reshape(1, -1),
            document_ids=[request.document_id],
            sources=[request.source],
            metadata=[json.dumps(request.metadata)]
        )
        
        if success:
            message = f"Document '{request.document_id}' uploaded successfully"
        else:
            message = f"Failed to upload document '{request.document_id}'"
        
        return DocumentUploadResponse(
            success=success,
            message=message,
            document_id=request.document_id
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")


@app.get("/system/info", response_model=SystemInfoResponse)
async def get_system_info():
    """Get comprehensive system information."""
    try:
        # Get model information
        llm_info = llm_manager.get_model_info()
        embedding_info = embedding_manager.get_model_info()
        classifier_info = query_classifier.get_model_info()
        
        # Get database information
        db_stats = await milvus_manager.get_collection_stats()
        
        # Get retrieval information
        retrieval_stats = await document_retriever.get_retrieval_stats()
        
        # Get workflow information
        workflow_info = await chatbot_workflow.get_workflow_info()
        
        return SystemInfoResponse(
            app_name=settings.APP_NAME,
            app_version=settings.APP_VERSION,
            models={
                "llm": llm_info,
                "embedding": embedding_info,
                "classifier": classifier_info
            },
            database=db_stats,
            retrieval=retrieval_stats,
            classification=classifier_info
        )
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting system info: {str(e)}")


@app.delete("/documents/clear")
async def clear_documents():
    """Clear all documents from the knowledge base."""
    try:
        success = await milvus_manager.clear_collection()
        
        if success:
            return {"success": True, "message": "All documents cleared successfully"}
        else:
            return {"success": False, "message": "Failed to clear documents"}
            
    except Exception as e:
        logger.error(f"Error clearing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")


@app.get("/documents/stats")
async def get_document_stats():
    """Get document statistics."""
    try:
        stats = await milvus_manager.get_collection_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting document stats: {str(e)}")


# Background task to initialize models
async def initialize_models_background():
    """Initialize models in the background."""
    try:
        logger.info("Starting background model initialization...")
        
        # Load embedding model first (needed for classification and retrieval)
        if not embedding_manager.is_loaded():
            await embedding_manager.load_model()
        
        # Train classifier
        if not query_classifier.is_trained:
            await query_classifier.train_classifier()
        
        # Load LLM model (this might take longer)
        if not llm_manager.is_loaded():
            await llm_manager.load_model()
        
        logger.info("Background model initialization completed")
        
    except Exception as e:
        logger.error(f"Error in background model initialization: {str(e)}")


@app.post("/system/initialize")
async def initialize_system(background_tasks: BackgroundTasks):
    """Initialize all system components."""
    background_tasks.add_task(initialize_models_background)
    return {"message": "System initialization started in background"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
