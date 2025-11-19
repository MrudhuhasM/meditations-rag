from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List

from meditations_rag.api.schemas.rag import QueryRequest, BatchQueryRequest
from meditations_rag.pipelines.rag import RAGResponse
from meditations_rag.api.dependencies import RagPipelineDep
from meditations_rag.config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/query", response_model=RAGResponse)
async def query_rag(
    request: QueryRequest,
    pipeline: RagPipelineDep
):
    """
    Execute a single RAG query against the Meditations knowledge base.
    """
    try:
        response = await pipeline.query(
            query=request.query,
            top_k=request.top_k,
            include_sources=request.include_sources,
            include_metadata=request.include_metadata
        )
        return response
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-query", response_model=List[RAGResponse])
async def batch_query_rag(
    request: BatchQueryRequest,
    pipeline: RagPipelineDep
):
    """
    Execute multiple RAG queries in parallel.
    """
    try:
        responses = await pipeline.batch_query(
            queries=request.queries,
            top_k=request.top_k,
            include_sources=request.include_sources,
            include_metadata=request.include_metadata,
            max_concurrent=request.max_concurrent
        )
        return responses
    except Exception as e:
        logger.error(f"Error processing batch RAG query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
