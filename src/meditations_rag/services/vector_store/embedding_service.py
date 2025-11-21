"""
Embedding Generation Service for Vector Store Upsertion.

Handles:
- Chunk text embedding generation
- Question text embedding generation
- Batch processing with concurrency control
- Integration with embedding factory

Design Principles:
- Single Responsibility: Only handles embedding generation for upsertion
- Dependency Inversion: Uses embedding provider abstractions
- Separation of Concerns: Distinct from chunk embedding service (which is for chunking)
"""

from typing import List

from meditations_rag.config import get_logger, settings
from meditations_rag.core.embedding.base import EmbeddingBase

logger = get_logger(__name__)


class VectorEmbeddingService:
    """
    Service for generating embeddings for vector store upsertion.

    This is separate from chunk embedding (used during chunking) and
    focuses on preparing embeddings for Qdrant upsertion.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingBase,
        batch_size: int | None = None,
        max_concurrent: int | None = None,
    ):
        """
        Initialize vector embedding service.

        Args:
            embedding_provider: Embedding provider instance
            batch_size: Batch size for embedding generation
            max_concurrent: Max concurrent embedding requests
        """
        self.embedding_provider = embedding_provider
        self.batch_size = batch_size or settings.rag.chunk_embed_batch_size
        self.max_concurrent = max_concurrent or settings.rag.max_concurrent_requests

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts")

        try:
            # Use the embedding provider to generate embeddings
            embeddings = await self.embedding_provider.embed_texts(texts)

            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise ValueError(f"Embedding generation failed: {e}") from e

    async def embed_chunks(self, chunks: List) -> List[List[float]]:
        """
        Generate embeddings for chunk texts.

        Args:
            chunks: List of llama_index BaseNode chunks

        Returns:
            List of embedding vectors
        """
        chunk_texts = [chunk.get_content() for chunk in chunks]
        logger.info(f"Embedding {len(chunk_texts)} chunk texts")

        return await self.embed_texts(chunk_texts)

    async def embed_questions_by_chunk(self, chunks: List) -> List[List[List[float]]]:
        """
        Generate embeddings for all questions from chunks.

        Returns a nested structure: for each chunk, a list of vectors
        for its questions.

        Args:
            chunks: List of llama_index BaseNode chunks with 'questions' in metadata

        Returns:
            List of lists - outer list per chunk, inner list contains vectors
            for that chunk's questions
        """
        logger.info(f"Embedding questions from {len(chunks)} chunks")

        # Collect all questions with their chunk index
        all_questions = []
        question_to_chunk_idx = []

        for chunk_idx, chunk in enumerate(chunks):
            questions = chunk.metadata.get("questions", [])

            for question in questions:
                all_questions.append(question)
                question_to_chunk_idx.append(chunk_idx)

        logger.info(f"Total questions to embed: {len(all_questions)}")

        if not all_questions:
            logger.warning("No questions found in chunks")
            return [[] for _ in chunks]

        # Generate embeddings for all questions at once
        question_embeddings = await self.embed_texts(all_questions)

        # Reorganize embeddings by chunk
        embeddings_by_chunk: List[List[List[float]]] = [[] for _ in chunks]

        for question_emb, chunk_idx in zip(question_embeddings, question_to_chunk_idx):
            embeddings_by_chunk[chunk_idx].append(question_emb)

        # Log statistics
        chunks_with_questions = sum(1 for embs in embeddings_by_chunk if embs)
        logger.info(
            f"Embedded questions for {chunks_with_questions}/{len(chunks)} chunks. "
            f"Total question embeddings: {len(question_embeddings)}"
        )

        return embeddings_by_chunk
