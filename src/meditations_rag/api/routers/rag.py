from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
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

@router.get("/stream-query")
async def stream_query_rag(
    query: str,
    pipeline: RagPipelineDep
):
    """
    Stream RAG query progress and result using Server-Sent Events (SSE).
    """
    async def event_generator():
        async for event in pipeline.stream_query_events(query=query):
            yield f"data: {event}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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
