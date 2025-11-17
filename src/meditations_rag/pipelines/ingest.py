from meditations_rag.services.loader import DocumentLoaderService
from meditations_rag.services.chunker import ChunkerService
from meditations_rag.config import get_logger
import asyncio


logger = get_logger(__name__)


class IngestPipeline:
    def __init__(self, loader: DocumentLoaderService, chunk_service: ChunkerService):
        self.loader = loader
        self.chunk_service = chunk_service

    async def ingest(self, file_path: str):
        start_time = asyncio.get_event_loop().time()
        logger.info(f"Starting ingestion for file: {file_path}")
        documents = await self.loader.load_documents(file_path)
        chunks = await self.chunk_service.chunk_documents(documents)
        end_time = asyncio.get_event_loop().time()
        logger.info(f"Completed ingestion for file: {file_path} in {end_time - start_time:.2f} seconds")
        logger.info(f"Total documents ingested: {len(documents)}")
        logger.info(f"Total chunks created: {len(chunks)}")
        return chunks