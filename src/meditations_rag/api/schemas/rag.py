from typing import Optional, List
from pydantic import BaseModel, Field
from meditations_rag.pipelines.rag import RAGResponse

class QueryRequest(BaseModel):
    """
    Request model for RAG query.
    """
    query: str = Field(..., description="The user's question about Meditations")
    top_k: Optional[int] = Field(None, description="Number of results to retrieve (overrides default)")
    include_sources: bool = Field(True, description="Whether to include source passages in the response")
    include_metadata: bool = Field(True, description="Whether to include metadata in the response")

class BatchQueryRequest(BaseModel):
    """
    Request model for batch RAG queries.
    """
    queries: List[str] = Field(..., description="List of user questions")
    top_k: Optional[int] = Field(None, description="Number of results to retrieve")
    include_sources: bool = Field(True, description="Whether to include source passages")
    include_metadata: bool = Field(True, description="Whether to include metadata")
    max_concurrent: int = Field(3, description="Maximum concurrent queries")

class HealthResponse(BaseModel):
    """
    Response model for health check.
    """
    status: str = Field(..., description="Status of the service (e.g., 'ok', 'unhealthy')")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Current environment")
