"""
Main entry point for the Meditations RAG ingestion pipeline.

This script orchestrates the loading, chunking, metadata extraction,
embedding, and upsertion of the Meditations text into the vector store.
"""

import asyncio
import os

from meditations_rag.config import get_logger, settings
from meditations_rag.core.chunk_embedding import get_chunk_embedding_model
from meditations_rag.core.embedding import create_embedding
from meditations_rag.core.llm import create_llm
from meditations_rag.pipelines.ingest import IngestPipeline
from meditations_rag.services.chunker import ChunkerService
from meditations_rag.services.loader import DocumentLoaderService
from meditations_rag.services.metadata import MetadataExtractorService
from meditations_rag.services.vector_store import (
    QdrantVectorStore,
    VectorEmbeddingService,
)

logger = get_logger(__name__)


async def main():
    """
    Run the ingestion pipeline.
    """
    # Construct path relative to the project root
    # Assuming this script is in src/meditations_rag/main.py
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    source_file = os.path.join(project_root, "data", "Marcus-Aurelius-Meditations.pdf")

    if not os.path.exists(source_file):
        logger.error(f"Source file not found: {source_file}")
        # Fallback to absolute path if relative fails (for dev environment)
        source_file = "C:\\Users\\mrudh\\Documents\\Projects\\ProfileProject\\Agentic-RAG\\meditations-rag\\data\\Marcus-Aurelius-Meditations.pdf"
        if not os.path.exists(source_file):
            logger.error(f"Source file still not found: {source_file}")
            return

    logger.info(f"Starting ingestion for: {source_file}")

    # Initialize services
    loader = DocumentLoaderService()
    chunk_embedding_model = get_chunk_embedding_model()
    chunk_service = ChunkerService(embed_model=chunk_embedding_model)
    embedding_base = create_embedding()
    embedding_service = VectorEmbeddingService(embedding_provider=embedding_base)
    vector_store = QdrantVectorStore()

    # Initialize LLM and Metadata Extractor
    llm = create_llm()
    metadata_extractor = MetadataExtractorService(
        llm=llm,
        batch_size=settings.rag.metadata_batch_size,
        max_concurrent=settings.rag.metadata_max_concurrent,
    )

    # Initialize Pipeline
    ingest_pipeline = IngestPipeline(
        loader=loader,
        chunk_service=chunk_service,
        metadata_extractor=metadata_extractor,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    try:
        chunk_results, question_results = await ingest_pipeline.ingest(
            file_path=source_file
        )

        logger.info(f"Chunk Ingestion Results: {chunk_results}")
        logger.info(f"Question Chunk Ingestion Results: {question_results}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
    finally:
        vector_store.close()


if __name__ == "__main__":
    asyncio.run(main())
