from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any
import json

from meditations_rag.api.schemas.rag import QueryRequest
from meditations_rag.api.dependencies import AgenticGraphDep
from meditations_rag.config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/agentic", tags=["Agentic RAG"])

@router.get("/stream-query")
async def stream_query_agentic(
    query: str,
    graph: AgenticGraphDep
):
    """
    Stream Agentic RAG query progress and result using Server-Sent Events (SSE).
    """
    async def event_generator():
        initial_state = {
            "query": query,
            "messages": [],
            "retrieved_docs": [],
            "search_queries": [],
            "iteration": 0,
            "active_model": "fast"
        }
        
        yield f"data: {json.dumps({'type': 'status', 'step': 'start', 'message': 'Starting Agentic Workflow...'})}\n\n"

        try:
            # Stream events from the graph
            # We use astream_events to get granular updates
            async for event in graph.astream_events(initial_state, version="v1"):
                kind = event["event"]
                name = event["name"]
                
                if kind == "on_chain_start" and name in ["controller", "retriever", "generator", "evaluator"]:
                    yield f"data: {json.dumps({'type': 'status', 'step': name, 'message': f'Running {name}...'})}\n\n"
                
                elif kind == "on_chain_end" and name == "LangGraph":
                    # Final result
                    final_state = event["data"]["output"]
                    
                    # Format response similar to RAGResponse
                    response = {
                        "query": query,
                        "answer": final_state.get("answer"),
                        "steps": {
                            "iterations": final_state.get("iteration"),
                            "search_queries": final_state.get("search_queries"),
                            "final_decision": final_state.get("decision"),
                            "evaluation_status": final_state.get("evaluation_status")
                        }
                    }
                    
                    # Format sources
                    sources = []
                    for doc in final_state.get("retrieved_docs", []):
                        source = {
                            "text": doc.text,
                            "chunk_id": doc.chunk_id,
                            "score": doc.score
                        }
                        if doc.metadata:
                            source.update(doc.metadata)
                        sources.append(source)
                    response["sources"] = sources
                    
                    yield f"data: {json.dumps({'type': 'result', 'data': response})}\n\n"

        except Exception as e:
            logger.error(f"Error in agentic stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/query")
async def query_agentic(
    request: QueryRequest,
    graph: AgenticGraphDep
):
    """
    Execute a query using the Agentic RAG pipeline (LangGraph).
    """
    try:
        # Initialize state
        initial_state = {
            "query": request.query,
            "messages": [],
            "retrieved_docs": [],
            "search_queries": [],
            "iteration": 0,
            "active_model": "fast"
        }
        
        # Run the graph
        # Note: ainvoke returns the final state
        final_state = await graph.ainvoke(initial_state)
        
        # Extract relevant information for response
        # We might want to define a specific response schema for Agentic RAG
        # that includes the reasoning steps (decision, search queries, etc.)
        
        response = {
            "query": request.query,
            "answer": final_state.get("answer"),
            "steps": {
                "iterations": final_state.get("iteration"),
                "search_queries": final_state.get("search_queries"),
                "final_decision": final_state.get("decision"),
                "evaluation_status": final_state.get("evaluation_status")
            }
        }
        
        if request.include_sources:
            # Format sources from retrieved_docs
            sources = []
            for doc in final_state.get("retrieved_docs", []):
                source = {
                    "text": doc.text,
                    "chunk_id": doc.chunk_id,
                    "score": doc.score
                }
                if request.include_metadata and doc.metadata:
                    source.update(doc.metadata)
                sources.append(source)
            response["sources"] = sources
            
        return response

    except Exception as e:
        logger.error(f"Error processing Agentic RAG query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
