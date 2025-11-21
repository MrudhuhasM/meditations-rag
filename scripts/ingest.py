# ruff: noqa: E402
import asyncio
import os
import sys
from pathlib import Path

# Add src to python path if running from root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))  # noqa: E402

from meditations_rag.config import get_logger  # noqa: E402
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
    # 1. Configuration
    file_path = "data/Marcus-Aurelius-Meditations.pdf"

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        # Fallback to checking absolute path if running from different cwd
        file_path = str(project_root / "data" / "Marcus-Aurelius-Meditations.pdf")
        if not os.path.exists(file_path):
            logger.error(f"File still not found: {file_path}")
            return

    logger.info(f"Starting ingestion for: {file_path}")

    # 2. Initialize Services
    try:
        # LLM for metadata extraction
        llm = create_llm()

        # Embedding model for vector store and chunking
        embedding_provider = create_embedding()
        chunk_embedding_model = get_chunk_embedding_model()

        # Services
        loader = DocumentLoaderService()
        chunker = ChunkerService(embed_model=chunk_embedding_model)
        metadata_extractor = MetadataExtractorService(llm=llm)
        vector_store = QdrantVectorStore()
        embedding_service = VectorEmbeddingService(
            embedding_provider=embedding_provider
        )

        # 3. Initialize Pipeline
        pipeline = IngestPipeline(
            loader=loader,
            chunk_service=chunker,
            metadata_extractor=metadata_extractor,
            vector_store=vector_store,
            embedding_service=embedding_service,
        )

        # 4. Run Ingestion
        await pipeline.ingest(file_path)

        logger.info("Ingestion completed successfully!")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
