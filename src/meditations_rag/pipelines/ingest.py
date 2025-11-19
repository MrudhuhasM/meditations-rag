from meditations_rag.services.loader import DocumentLoaderService
from meditations_rag.services.chunker import ChunkerService
from meditations_rag.services.metadata import MetadataExtractorService, ChunkMetadata
from meditations_rag.services.vector_store import QdrantVectorStore, VectorEmbeddingService
from meditations_rag.config import get_logger, settings
import asyncio


logger = get_logger(__name__)


class IngestPipeline:
    """
    Document ingestion pipeline with optional metadata extraction and vector store upsertion.
    
    Orchestrates: loading -> chunking -> metadata extraction -> embedding -> upsertion
    
    Design Principles:
    - Single Responsibility: Coordinates ingestion workflow
    - Dependency Inversion: Depends on service abstractions
    - Open/Closed: Extensible via service injection
    """
    
    def __init__(
        self,
        loader: DocumentLoaderService,
        chunk_service: ChunkerService,
        metadata_extractor: MetadataExtractorService,
        vector_store: QdrantVectorStore,
        embedding_service: VectorEmbeddingService
    ):
        self.loader = loader
        self.chunk_service = chunk_service
        self.metadata_extractor = metadata_extractor
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def _enrich_chunk_metadata(self, chunk, extracted_metadata: ChunkMetadata) -> None:
        """
        Enrich chunk's existing metadata with extracted structured metadata.
        
        Adds extracted metadata fields to the chunk's metadata dict without
        overwriting existing metadata (file_path, source, total_pages, etc.).
        
        Args:
            chunk: llama_index TextNode/BaseNode object
            extracted_metadata: Extracted ChunkMetadata from LLM
        """
        # Add extracted metadata to chunk's metadata dict
        chunk.metadata["questions"] = extracted_metadata.questions
        chunk.metadata["keywords"] = extracted_metadata.keywords
        chunk.metadata["topic"] = extracted_metadata.topic.value  # Store enum value as string
        chunk.metadata["entities"] = extracted_metadata.entities
        chunk.metadata["philosophical_concepts"] = extracted_metadata.philosophical_concepts
        chunk.metadata["stoic_practices"] = extracted_metadata.stoic_practices

    async def ingest(self, file_path: str, upsert_to_vector_store: bool = True):
        start_time = asyncio.get_event_loop().time()
        logger.info(f"Starting ingestion for file: {file_path}")
        

        documents = await self.loader.load_documents(file_path)
        logger.info(f"Loaded {len(documents)} documents")
        

        chunks = self.chunk_service.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        

        if self.metadata_extractor and settings.rag.metadata_extraction_enabled:
            logger.info("Starting metadata extraction...")
            chunk_texts = [chunk.get_content() for chunk in chunks]
            chunk_metadata_list = await self.metadata_extractor.extract_batch_metadata(chunk_texts)
            
            
            enriched_count = 0
            for chunk, extracted_metadata in zip(chunks, chunk_metadata_list):
                if extracted_metadata is not None:
                    self._enrich_chunk_metadata(chunk, extracted_metadata)
                    enriched_count += 1
                else:
                    logger.warning(f"Chunk {chunk.node_id} metadata extraction failed, keeping original metadata")
            
            logger.info(f"Enriched {enriched_count}/{len(chunks)} chunks with extracted metadata")
        
        
        if upsert_to_vector_store and self.vector_store and self.embedding_service:
            logger.info("Starting vector store upsertion...")
            
        
            self.vector_store.ensure_collections_exist()
            
        
            logger.info("Generating chunk embeddings...")
            chunk_vectors = await self.embedding_service.embed_chunks(chunks)
            
        
            logger.info("Upserting chunks to main collection...")
            chunk_results = await self.vector_store.upsert_chunks_batch(chunks, chunk_vectors)
            logger.info(
                f"Chunk upsertion: {chunk_results['successful']}/{chunk_results['total']} "
                f"successful ({chunk_results['success_rate']:.1f}%)"
            )
            
        
            logger.info("Generating question embeddings...")
            question_vectors_by_chunk = await self.embedding_service.embed_questions_by_chunk(chunks)
            
            logger.info("Upserting questions to questions collection...")
            question_results = await self.vector_store.upsert_questions_batch(
                chunks, question_vectors_by_chunk
            )
            logger.info(
                f"Question upsertion: {question_results['successful']}/{question_results['total']} "
                f"successful ({question_results['success_rate']:.1f}%)"
            )
        
        end_time = asyncio.get_event_loop().time()
        logger.info(f"Completed ingestion for file: {file_path} in {end_time - start_time:.2f} seconds")
        logger.info(f"Total documents ingested: {len(documents)}")
        logger.info(f"Total chunks created: {len(chunks)}")

        return chunk_results, question_results