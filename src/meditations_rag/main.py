from meditations_rag.services.loader import DocumentLoaderService
from meditations_rag.services.chunker import ChunkerService
from meditations_rag.services.metadata import MetadataExtractorService
from meditations_rag.core.chunk_embeding import get_chunk_embedding_model
from meditations_rag.core.llm import create_llm
from meditations_rag.pipelines.ingest import IngestPipeline
from meditations_rag.config import settings


async def main():
    source_file = "C:\\Users\\mrudh\\Documents\\Projects\\ProfileProject\\Agentic-RAG\\meditations-rag\\data\\Marcus-Aurelius-Meditations.pdf"

        # Initialize services
    loader = DocumentLoaderService()
    chunk_embedding_model = get_chunk_embedding_model()
    chunk_service = ChunkerService(embed_model=chunk_embedding_model)

    # Create metadata extractor
    llm = create_llm()
    metadata_extractor = MetadataExtractorService(llm=llm, batch_size=5, max_concurrent=3)

    # Create pipeline with metadata extraction
    ingest_pipeline = IngestPipeline(
        loader=loader, 
        chunk_service=chunk_service,
        metadata_extractor=metadata_extractor
    )
    
    
    chunks = await ingest_pipeline.ingest(file_path=source_file)
    
    print(f"\n{'='*80}")
    print(f"Ingestion Complete")
    print(f"{'='*80}")
    print(f"Total chunks created: {len(chunks)}")
    
    # Show sample metadata if extraction was enabled
    if settings.rag.metadata_extraction_enabled and len(chunks) > 0:
        sample_chunk = chunks[0]
        if "topic" in sample_chunk.metadata:
            print(f"\nSample Enriched Metadata:")
            print(f"  Topic: {sample_chunk.metadata.get('topic')}")
            print(f"  Questions: {len(sample_chunk.metadata.get('questions', []))}")
            print(f"  Keywords: {len(sample_chunk.metadata.get('keywords', []))}")
            print(f"  Entities: {len(sample_chunk.metadata.get('entities', []))}")
            print(f"  Extracted at: {sample_chunk.metadata.get('metadata_extracted_at')}")
        else:
            print(f"\nMetadata extraction was enabled but no extracted metadata found in chunks")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
