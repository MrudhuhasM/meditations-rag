from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from meditations_rag.config import get_logger, settings
from meditations_rag.core.embedding.base import EmbeddingBase
from meditations_rag.core.exceptions import (
    EmbeddingAPIError,
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingDimensionMismatchError,
    EmbeddingRateLimitError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
)
from meditations_rag.core.rate_limiter import RateLimiterFactory, rate_limit

logger = get_logger(__name__)


class OpenAIEmbedding(EmbeddingBase):
    """OpenAI embedding implementation."""

    def __init__(self):
        self.settings = settings.openai

        if not self.settings.api_key:
            raise EmbeddingConfigurationError(
                "OpenAI API key is not configured. Please set OPENAI_API_KEY in your environment.",
                details={"provider": "openai"},
            )

        try:
            self.client = AsyncOpenAI(api_key=self.settings.api_key.get_secret_value())
            self.model_name = self.settings.embedding_model_name
            logger.info(f"Initialized OpenAIEmbedding with model: {self.model_name}")
        except Exception as e:
            raise EmbeddingConfigurationError(
                f"Failed to initialize OpenAI embedding client: {str(e)}",
                details={"provider": "openai", "error": str(e)},
            )

    @rate_limit(RateLimiterFactory.get_openai_limiter)
    @retry(
        stop=stop_after_attempt(settings.openai.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
    )
    async def embed_text(self, text: str) -> list[float]:
        """Generate embeddings for a single text."""
        try:
            if not text or not text.strip():
                logger.warning("Attempting to embed empty text")
                raise EmbeddingResponseError(
                    "Cannot embed empty or whitespace-only text",
                    provider="openai",
                    details={"model": self.model_name},
                )

            logger.debug(f"Generating embedding for text of length: {len(text)}")
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text,
                dimensions=settings.rag.embedding_dimension,
            )

            if not response.data or len(response.data) == 0:
                logger.error("OpenAI embeddings API returned no data")
                raise EmbeddingResponseError(
                    "OpenAI embeddings API returned no data",
                    provider="openai",
                    expected_count=1,
                    received_count=0,
                    details={"model": self.model_name},
                )

            embedding = response.data[0].embedding

            # Validate embedding dimension
            if len(embedding) != settings.rag.embedding_dimension:
                logger.error(
                    f"Embedding dimension mismatch: expected {settings.rag.embedding_dimension}, got {len(embedding)}"
                )
                raise EmbeddingDimensionMismatchError(
                    "Embedding dimension does not match expected dimension",
                    expected_dimension=settings.rag.embedding_dimension,
                    received_dimension=len(embedding),
                    details={"model": self.model_name},
                )

            logger.debug(
                f"Successfully generated embedding with dimension {len(embedding)}"
            )
            return embedding

        except AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {str(e)}")
            raise EmbeddingAuthenticationError(
                "OpenAI authentication failed. Please verify your API key is correct and active.",
                provider="openai",
                details={"error": str(e), "model": self.model_name},
            ) from e

        except RateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded: {str(e)}")
            raise EmbeddingRateLimitError(
                "OpenAI rate limit exceeded. Please wait before retrying.",
                provider="openai",
                details={
                    "error": str(e),
                    "model": self.model_name,
                    "max_retries": settings.openai.max_retries,
                },
            ) from e

        except APITimeoutError as e:
            logger.error(f"OpenAI API timeout: {str(e)}")
            raise EmbeddingTimeoutError(
                "OpenAI embeddings API request timed out",
                provider="openai",
                details={"error": str(e), "model": self.model_name},
            ) from e

        except APIConnectionError as e:
            logger.error(f"OpenAI API connection error: {str(e)}")
            raise EmbeddingAPIError(
                "Failed to connect to OpenAI embeddings API. Check your network connection.",
                provider="openai",
                model=self.model_name,
                text_count=1,
                error_type="connection_error",
                details={"error": str(e)},
            ) from e

        except APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise EmbeddingAPIError(
                f"OpenAI embeddings API error: {str(e)}",
                provider="openai",
                model=self.model_name,
                text_count=1,
                status_code=getattr(e, "status_code", None),
                error_type=getattr(e, "type", None),
                details={"error": str(e)},
            ) from e

        except (EmbeddingResponseError, EmbeddingDimensionMismatchError):
            # Re-raise our custom exceptions
            raise

        except Exception as e:
            logger.error(
                f"Unexpected error during OpenAI embedding: {str(e)}", exc_info=True
            )
            raise EmbeddingAPIError(
                f"Unexpected error during OpenAI embedding generation: {str(e)}",
                provider="openai",
                model=self.model_name,
                text_count=1,
                error_type="unexpected_error",
                details={"error": str(e), "error_type": type(e).__name__},
            ) from e

    # embed_texts is now handled by the base class

    @rate_limit(RateLimiterFactory.get_openai_limiter)
    @retry(
        stop=stop_after_attempt(settings.openai.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a single batch (internal method with rate limiting and retry)."""
        logger.debug(f"Generating embeddings for batch of {len(texts)} texts")
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            dimensions=settings.rag.embedding_dimension,
        )

        if not response.data or len(response.data) == 0:
            logger.error("OpenAI embeddings API returned no data for batch")
            raise EmbeddingResponseError(
                "OpenAI embeddings API returned no data for batch",
                provider="openai",
                expected_count=len(texts),
                received_count=0,
                details={"model": self.model_name},
            )

        if len(response.data) != len(texts):
            logger.error(
                f"Embedding count mismatch: expected {len(texts)}, got {len(response.data)}"
            )
            raise EmbeddingResponseError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(response.data)}",
                provider="openai",
                expected_count=len(texts),
                received_count=len(response.data),
                details={"model": self.model_name},
            )

        # Ensure results are in the same order as input and validate dimensions
        embeddings = []
        for i, item in enumerate(response.data):
            if len(item.embedding) != settings.rag.embedding_dimension:
                logger.error(
                    f"Embedding {i} dimension mismatch: expected {settings.rag.embedding_dimension}, got {len(item.embedding)}"
                )
                raise EmbeddingDimensionMismatchError(
                    f"Embedding dimension does not match expected dimension at index {i}",
                    expected_dimension=settings.rag.embedding_dimension,
                    received_dimension=len(item.embedding),
                    details={"model": self.model_name, "index": i},
                )
            embeddings.append(item.embedding)

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return settings.rag.embedding_dimension
