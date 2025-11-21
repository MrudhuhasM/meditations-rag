from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends
from meditations_rag.config import get_logger, settings
from meditations_rag.core.embedding import create_embedding
from meditations_rag.core.llm import create_llm
from meditations_rag.pipelines.agentic_rag import create_agentic_rag_graph
from meditations_rag.pipelines.rag import RAGPipeline, create_rag_pipeline
from meditations_rag.services.retrieval import QueryRewriter, RetrievalService
from meditations_rag.services.vector_store import (
    QdrantVectorStore,
    VectorEmbeddingService,
)

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    """
    Dependency that provides a singleton instance of the RAGPipeline.
    """
    logger.info("Initializing RAG Pipeline for API")

    llm = create_llm()
    embedding_provider = create_embedding()

    pipeline = create_rag_pipeline(
        llm=llm,
        embedding_provider=embedding_provider,
        enable_query_rewriting=True,  # Can be configurable via settings
        enable_metadata_retrieval=settings.rag.retrieval_metadata_enabled,
    )

    return pipeline


@lru_cache(maxsize=1)
def get_agentic_graph():
    """
    Dependency that provides the compiled Agentic RAG graph.
    """
    logger.info("Initializing Agentic RAG Graph for API")

    # We need to reconstruct the components for the graph
    # Ideally, we should refactor RAGPipeline to expose these, but for now we'll recreate them
    # or reuse if possible.

    llm_fast = create_llm()  # Default LLM

    # Check if we have a strong LLM configured, otherwise use the same one
    # For now, we'll just use the same one or create a new one if settings differ
    # In a real scenario, you might want to pass a specific provider name to create_llm
    llm_strong = create_llm()

    embedding_provider = create_embedding()
    vector_store = QdrantVectorStore()
    embedding_service = VectorEmbeddingService(embedding_provider=embedding_provider)

    query_rewriter = QueryRewriter(llm=llm_fast)

    retrieval_service = RetrievalService(
        vector_store=vector_store,
        embedding_service=embedding_service,
        query_rewriter=query_rewriter,
        alpha=settings.rag.retrieval_alpha,
        top_k=settings.rag.retrieval_top_k,
        enable_metadata_retrieval=settings.rag.retrieval_metadata_enabled,
    )

    graph = create_agentic_rag_graph(
        retrieval_service=retrieval_service, llm_fast=llm_fast, llm_strong=llm_strong
    )

    return graph


RagPipelineDep = Annotated[RAGPipeline, Depends(get_rag_pipeline)]
AgenticGraphDep = Annotated[Any, Depends(get_agentic_graph)]
