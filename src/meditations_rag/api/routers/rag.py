from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from meditations_rag.api.dependencies import RagPipelineDep
from meditations_rag.api.limiter import limiter
from meditations_rag.api.schemas.rag import BatchQueryRequest, QueryRequest
from meditations_rag.api.security import verify_standard_rag_access
from meditations_rag.config import get_logger
from meditations_rag.pipelines.rag import RAGResponse
from meditations_rag.services.input_validator import input_validator

logger = get_logger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post(
    "/query",
    response_model=RAGResponse,
    dependencies=[Depends(verify_standard_rag_access)],
)
@limiter.limit("20/minute")
async def query_rag(
    request: Request, query_request: QueryRequest, pipeline: RagPipelineDep
):
    """
    Execute a single RAG query against the Meditations knowledge base.
    """
    if not input_validator.validate(query_request.query):
        raise HTTPException(status_code=400, detail="No")

    try:
        response = await pipeline.query(
            query=query_request.query,
            top_k=query_request.top_k,
            include_sources=query_request.include_sources,
            include_metadata=query_request.include_metadata,
        )
        return response
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream-query", dependencies=[Depends(verify_standard_rag_access)])
@limiter.limit("20/minute")
async def stream_query_rag(request: Request, query: str, pipeline: RagPipelineDep):
    """
    Stream RAG query progress and result using Server-Sent Events (SSE).
    """
    if not input_validator.validate(query):
        raise HTTPException(status_code=400, detail="No")

    async def event_generator():
        async for event in pipeline.stream_query_events(query=query):
            yield f"data: {event}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post(
    "/batch-query",
    response_model=List[RAGResponse],
    dependencies=[Depends(verify_standard_rag_access)],
)
@limiter.limit("5/minute")
async def batch_query_rag(
    request: Request, batch_request: BatchQueryRequest, pipeline: RagPipelineDep
):
    """
    Execute multiple RAG queries in parallel.
    """
    for q in batch_request.queries:
        if not input_validator.validate(q):
            raise HTTPException(status_code=400, detail="No")

    try:
        responses = await pipeline.batch_query(
            queries=batch_request.queries,
            top_k=batch_request.top_k,
            include_sources=batch_request.include_sources,
            include_metadata=batch_request.include_metadata,
            max_concurrent=batch_request.max_concurrent,
        )
        return responses
    except Exception as e:
        logger.error(f"Error processing batch RAG query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
