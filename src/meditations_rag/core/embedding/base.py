from abc import ABC, abstractmethod

from meditations_rag.config import get_logger, settings

logger = get_logger(__name__)


class EmbeddingBase(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embeddings for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts and log warning
        valid_texts = [t for t in texts if t and t.strip()]
        if len(valid_texts) < len(texts):
            logger.warning(
                f"Filtered out {len(texts) - len(valid_texts)} empty texts from batch"
            )
            if not valid_texts:
                return []

        # Split into batches if needed (API limit is typically 32)
        api_batch_size = settings.rag.embedding_api_batch_size
        all_embeddings = []

        if len(valid_texts) <= api_batch_size:
            # Single batch - process directly
            logger.debug(
                f"Generating embeddings for {len(valid_texts)} texts in single batch"
            )
            batch_embeddings = await self._embed_batch(valid_texts)
            all_embeddings.extend(batch_embeddings)
        else:
            # Multiple batches needed
            num_batches = (len(valid_texts) + api_batch_size - 1) // api_batch_size
            logger.info(
                f"Splitting {len(valid_texts)} texts into {num_batches} batches of max {api_batch_size}"
            )

            for i in range(0, len(valid_texts), api_batch_size):
                batch = valid_texts[i : i + api_batch_size]
                batch_num = (i // api_batch_size) + 1
                logger.debug(
                    f"Processing batch {batch_num}/{num_batches} with {len(batch)} texts"
                )

                batch_embeddings = await self._embed_batch(batch)
                all_embeddings.extend(batch_embeddings)

        logger.debug(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings

    @abstractmethod
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a single batch.
        To be implemented by providers.
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Integer dimension of embedding vectors
        """
        pass
