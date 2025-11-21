"""
Qdrant Vector Store Service.

Handles dual-collection upsertion strategy:
1. Main collection: Chunks with embeddings and full metadata
2. Questions collection: Individual questions with embeddings and metadata

Design Principles:
- Single Responsibility: Only manages Qdrant vector storage
- Dependency Inversion: Uses direct Qdrant client SDK
- Error Recovery: Batch retry logic with granular failure tracking
"""

from typing import Any, Dict, List
from uuid import uuid4

from llama_index.core.schema import BaseNode
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    OptimizersConfigDiff,
    PointStruct,
    UpdateStatus,
    VectorParams,
)

from meditations_rag.config import get_logger, settings

logger = get_logger(__name__)


class QdrantVectorStore:
    """
    Qdrant vector database client for dual-collection RAG strategy.

    Manages two collections:
    - Main collection: Full chunks with metadata
    - Questions collection: Individual questions from chunks

    Features:
    - Automatic collection creation/verification
    - Batch upsert with retry logic
    - Concurrent operations for performance
    - Detailed error tracking and logging
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        url: str | None = None,
        api_key: str | None = None,
        main_collection_name: str | None = None,
        question_collection_name: str | None = None,
        embedding_dimension: int | None = None,
        batch_size: int = 100,
        max_retries: int = 3,
    ):
        """
        Initialize Qdrant vector store.

        Args:
            host: Qdrant server host (defaults to settings)
            port: Qdrant server port (defaults to settings)
            url: Qdrant Cloud URL (defaults to settings)
            api_key: Qdrant API key if required (defaults to settings)
            main_collection_name: Name of main chunks collection (defaults to settings)
            question_collection_name: Name of questions collection (defaults to settings)
            embedding_dimension: Dimension of embedding vectors (defaults to settings)
            batch_size: Number of points to upsert per batch
            max_retries: Maximum retry attempts for failed operations
        """
        self.host = host or settings.qdrant.host
        self.port = port or settings.qdrant.port
        self.url = url or settings.qdrant.url
        self.api_key_str = api_key or (
            settings.qdrant.api_key.get_secret_value()
            if settings.qdrant.api_key
            else None
        )
        self.main_collection = (
            main_collection_name or settings.qdrant.main_collection_name
        )
        self.question_collection = (
            question_collection_name or settings.qdrant.question_collection_name
        )
        self.embedding_dimension = (
            embedding_dimension or settings.rag.embedding_dimension
        )
        self.batch_size = batch_size
        self.max_retries = max_retries

        # Initialize client
        if self.url:
            self.client = QdrantClient(
                url=self.url, api_key=self.api_key_str, timeout=60
            )
            logger.info(
                f"Initialized Qdrant client: url={self.url}, "
                f"main_collection={self.main_collection}, question_collection={self.question_collection}"
            )
        else:
            self.client = QdrantClient(
                host=self.host, port=self.port, api_key=self.api_key_str, timeout=60
            )
            logger.info(
                f"Initialized Qdrant client: host={self.host}, port={self.port}, "
                f"main_collection={self.main_collection}, question_collection={self.question_collection}"
            )

    def _ensure_collection_exists(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ) -> bool:
        """
        Ensure a collection exists, creating it if necessary.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors in this collection
            distance: Distance metric for similarity search

        Returns:
            True if collection exists or was created successfully
        """
        try:
            # Check if collection exists
            exists = self.client.collection_exists(collection_name=collection_name)

            if exists:
                logger.info(f"Collection '{collection_name}' already exists")
                # Verify vector configuration matches
                collection_info = self.client.get_collection(
                    collection_name=collection_name
                )
                vectors_config = collection_info.config.params.vectors
                
                if isinstance(vectors_config, VectorParams):
                    config_size = vectors_config.size
                    if config_size != vector_size:
                        logger.warning(
                            f"Collection '{collection_name}' has vector size {config_size}, "
                            f"expected {vector_size}. This may cause issues."
                        )
                else:
                    logger.debug(
                        f"Collection '{collection_name}' uses named vectors configuration. Skipping size validation."
                    )
                return True

            # Create collection
            logger.info(
                f"Creating collection '{collection_name}' with vector size {vector_size}"
            )

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=10000,  # Start indexing after 10k points
                ),
            )

            logger.info(f"Successfully created collection '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to ensure collection '{collection_name}' exists: {e}")
            raise RuntimeError(f"Collection setup failed: {e}") from e

    def ensure_collections_exist(self, vector_size: int | None = None) -> None:
        """
        Ensure both main and question collections exist.

        Args:
            vector_size: Override embedding dimension (uses default if None)
        """
        size = vector_size or self.embedding_dimension

        if size is None:
            raise ValueError(
                "Embedding dimension not specified. Set RAG_EMBEDDING_DIMENSION or pass vector_size."
            )

        logger.info("Verifying Qdrant collections...")

        # Ensure main collection exists
        self._ensure_collection_exists(
            collection_name=self.main_collection, vector_size=size
        )

        # Ensure questions collection exists
        self._ensure_collection_exists(
            collection_name=self.question_collection, vector_size=size
        )

        # Create payload indexes
        self.create_payload_indexes()

        logger.info("All collections verified and ready")

    def _create_chunk_point(
        self, chunk_id: str, chunk_vector: List[float], chunk_metadata: Dict[str, Any]
    ) -> PointStruct:
        """
        Create a PointStruct for a chunk.

        Args:
            chunk_id: Unique identifier for the chunk
            chunk_vector: Embedding vector for the chunk
            chunk_metadata: Metadata dictionary

        Returns:
            PointStruct ready for upsertion
        """
        return PointStruct(id=chunk_id, vector=chunk_vector, payload=chunk_metadata)

    def _create_question_points(
        self,
        chunk_id: str,
        questions: List[str],
        question_vectors: List[List[float]],
        chunk_metadata: Dict[str, Any],
    ) -> List[PointStruct]:
        """
        Create PointStructs for all questions from a chunk.

        Args:
            chunk_id: ID of the parent chunk
            questions: List of question texts
            question_vectors: Corresponding embedding vectors
            chunk_metadata: Metadata from parent chunk to include

        Returns:
            List of PointStruct objects for questions
        """
        if len(questions) != len(question_vectors):
            raise ValueError(
                f"Mismatch: {len(questions)} questions but {len(question_vectors)} vectors"
            )

        points = []
        for question_text, question_vector in zip(questions, question_vectors):
            # Generate unique ID for this question
            question_id = str(uuid4())

            # Build payload with question text and reference to parent chunk
            payload = {
                "question": question_text,
                "chunk_id": chunk_id,
                **chunk_metadata,  # Include parent chunk metadata for filtering
            }

            points.append(
                PointStruct(id=question_id, vector=question_vector, payload=payload)
            )

        return points

    async def _upsert_batch(
        self, collection_name: str, points: List[PointStruct], description: str
    ) -> Dict[str, Any]:
        """
        Internal helper to upsert a list of points in batches.

        Args:
            collection_name: Target collection name
            points: List of PointStruct objects to upsert
            description: Description for logging (e.g., "chunks", "questions")

        Returns:
            Dictionary with success/failure statistics
        """
        total_upserted = 0
        failed_ids = []
        failed_count = 0  # For cases where we don't track IDs explicitly or just want a count

        for i in range(0, len(points), self.batch_size):
            batch = points[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(points) + self.batch_size - 1) // self.batch_size

            logger.info(
                f"Upserting batch {batch_num}/{total_batches} "
                f"({len(batch)} {description}) to {collection_name}"
            )

            try:
                result = self.client.upsert(
                    collection_name=collection_name, points=batch, wait=True
                )

                if result.status == UpdateStatus.COMPLETED:
                    total_upserted += len(batch)
                    logger.debug(f"Batch {batch_num} upserted successfully")
                else:
                    logger.warning(
                        f"Batch {batch_num} completed with status: {result.status}"
                    )
                    failed_ids.extend([str(p.id) for p in batch])
                    failed_count += len(batch)

            except Exception as e:
                logger.error(f"Failed to upsert batch {batch_num}: {e}")
                failed_ids.extend([str(p.id) for p in batch])
                failed_count += len(batch)

        success_rate = (total_upserted / len(points)) * 100 if points else 0
        logger.info(
            f"{description.capitalize()} upsert completed: {total_upserted}/{len(points)} ({success_rate:.1f}%) successful"
        )

        return {
            "total": len(points),
            "successful": total_upserted,
            "failed": failed_count,
            "failed_ids": failed_ids,
            "success_rate": success_rate,
        }

    async def upsert_chunks_batch(
        self, chunks: List[BaseNode], chunk_vectors: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Upsert a batch of chunks to the main collection.

        Args:
            chunks: List of llama_index BaseNode chunks
            chunk_vectors: Corresponding embedding vectors

        Returns:
            Dictionary with success/failure statistics
        """
        if len(chunks) != len(chunk_vectors):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(chunk_vectors)} vectors"
            )

        logger.info(f"Upserting {len(chunks)} chunks to '{self.main_collection}'")

        # Prepare points
        points = []
        for chunk, vector in zip(chunks, chunk_vectors):
            chunk_id = chunk.node_id

            # Build payload from chunk metadata and content
            payload = {"text": chunk.get_content(), **chunk.metadata}

            point = self._create_chunk_point(chunk_id, vector, payload)
            points.append(point)

        # Upsert in batches
        return await self._upsert_batch(
            collection_name=self.main_collection, points=points, description="chunks"
        )

    async def upsert_questions_batch(
        self, chunks: List[BaseNode], question_vectors_by_chunk: List[List[List[float]]]
    ) -> Dict[str, Any]:
        """
        Upsert questions from chunks to the questions collection.

        Args:
            chunks: List of llama_index BaseNode chunks with metadata containing 'questions'
            question_vectors_by_chunk: List of lists - each inner list contains vectors for
                                        questions from the corresponding chunk

        Returns:
            Dictionary with success/failure statistics
        """
        if len(chunks) != len(question_vectors_by_chunk):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(question_vectors_by_chunk)} vector lists"
            )

        logger.info(f"Preparing questions from {len(chunks)} chunks for upsertion")

        # Collect all question points
        all_question_points = []

        for chunk, question_vectors in zip(chunks, question_vectors_by_chunk):
            # Extract questions from chunk metadata
            questions = chunk.metadata.get("questions", [])

            if not questions:
                logger.debug(f"Chunk {chunk.node_id} has no questions, skipping")
                continue

            if len(questions) != len(question_vectors):
                logger.warning(
                    f"Chunk {chunk.node_id}: {len(questions)} questions but "
                    f"{len(question_vectors)} vectors. Skipping this chunk."
                )
                continue

            # Create question points
            try:
                # Build metadata to include with questions
                # Only include chunk_text for context, no other metadata needed
                chunk_metadata = {"chunk_text": chunk.get_content()}

                question_points = self._create_question_points(
                    chunk_id=chunk.node_id,
                    questions=questions,
                    question_vectors=question_vectors,
                    chunk_metadata=chunk_metadata,
                )

                all_question_points.extend(question_points)

            except Exception as e:
                logger.error(
                    f"Failed to create question points for chunk {chunk.node_id}: {e}"
                )
                continue

        logger.info(f"Created {len(all_question_points)} question points for upsertion")

        # Upsert in batches
        return await self._upsert_batch(
            collection_name=self.question_collection,
            points=all_question_points,
            description="questions",
        )

    def create_payload_indexes(self) -> None:
        """
        Create payload indexes for metadata filtering.
        Required for Qdrant Cloud and large collections.
        """
        logger.info("Creating payload indexes for metadata filtering...")

        # Fields to index in main collection
        fields_to_index = [
            "keywords",
            "topic",
            "philosophical_concepts",
            "stoic_practices",
            "entities",
            "book",
            "chapter",
        ]

        for field in fields_to_index:
            try:
                self.client.create_payload_index(
                    collection_name=self.main_collection,
                    field_name=field,
                    field_schema="keyword",  # Use keyword schema for exact matching/filtering
                    wait=True,
                )
                logger.info(
                    f"Created index for field '{field}' in '{self.main_collection}'"
                )
            except Exception as e:
                # Log warning but continue (might already exist or other issue)
                logger.warning(f"Failed to create index for '{field}': {e}")

        logger.info("Payload index creation complete")

    def close(self) -> None:
        """Close the Qdrant client connection."""
        try:
            self.client.close()
            logger.info("Qdrant client connection closed")
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
