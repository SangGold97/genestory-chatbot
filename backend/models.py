from typing import List, Optional, Dict, Any
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


def numpy_json_encoder(obj):
    """Custom JSON encoder for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class BaseModelWithNumpySupport(BaseModel):
    """Base model with numpy type support."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            np.integer: lambda x: int(x),
            np.floating: lambda x: float(x),
            np.ndarray: lambda x: x.tolist(),
        }
    )


class QueryRequest(BaseModelWithNumpySupport):
    """Request model for chat queries."""
    query: str = Field(..., description="User query")
    conversation_history: Optional[List[str]] = Field(
        default=None, 
        description="Previous conversation turns"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class QueryResponse(BaseModelWithNumpySupport):
    """Response model for chat queries."""
    query: str
    response: str
    classification: Dict[str, Any]
    num_documents_retrieved: int
    num_documents_reranked: int
    metadata: Dict[str, Any]
    processing_time_ms: float


class DocumentUploadRequest(BaseModelWithNumpySupport):
    """Request model for document upload."""
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Document source")
    document_id: str = Field(..., description="Unique document ID")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Document metadata"
    )


class DocumentUploadResponse(BaseModelWithNumpySupport):
    """Response model for document upload."""
    success: bool
    message: str
    document_id: str


class ModelLoadRequest(BaseModelWithNumpySupport):
    """Request model for loading models."""
    model_type: str = Field(..., description="Type of model (llm, embedding)")


class ModelLoadResponse(BaseModelWithNumpySupport):
    """Response model for loading models."""
    success: bool
    message: str
    model_info: Dict[str, Any]


class HealthResponse(BaseModelWithNumpySupport):
    """Response model for health check."""
    status: str
    models: Dict[str, bool]
    database: Dict[str, Any]
    uptime_seconds: float


class SystemInfoResponse(BaseModelWithNumpySupport):
    """Response model for system information."""
    app_name: str
    app_version: str
    models: Dict[str, Any]
    database: Dict[str, Any]
    retrieval: Dict[str, Any]
    classification: Dict[str, Any]


class ErrorResponse(BaseModelWithNumpySupport):
    """Response model for errors."""
    error: str
    detail: Optional[str] = None
    status_code: int
