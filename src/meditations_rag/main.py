from meditations_rag.services.loader import DocumentLoaderService
from meditations_rag.services.chunker import ChunkerService
from meditations_rag.core.chunk_embedding import get_chunk_embedding_model
from meditations_rag.pipelines.ingest import IngestPipeline

async def main():
    source_file = "C:\\Users\\mrudh\\Documents\\Projects\\ProfileProject\\Agentic-RAG\\meditations-rag\\data\\Marcus-Aurelius-Meditations.pdf"
    loader = DocumentLoaderService()
    chunk_embedding_model = get_chunk_embedding_model()
    chunk_service = ChunkerService(embed_model=chunk_embedding_model)
    ingest_pipeline = IngestPipeline(loader=loader, chunk_service=chunk_service)
    chunks = await ingest_pipeline.ingest(file_path=source_file)
    print(f"Total chunks created: {len(chunks)}")
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())