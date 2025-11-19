from google import genai
from google.genai import types, errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from meditations_rag.config import settings, get_logger
from meditations_rag.core.embedding.base import EmbeddingBase
from meditations_rag.core.rate_limiter import rate_limit, RateLimiterFactory
from meditations_rag.core.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingAPIError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    EmbeddingResponseError,
    EmbeddingAuthenticationError,
    EmbeddingDimensionMismatchError,
)

logger = get_logger(__name__)


class GeminiEmbedding(EmbeddingBase):
    """Gemini embedding implementation."""
    
    def __init__(self):
        self.settings = settings.gemini

        if not self.settings.api_key:
            raise EmbeddingConfigurationError(
                'Gemini API key is not configured. Please set GEMINI_API_KEY in your environment.',
                details={'provider': 'gemini'}
            )

        try:
            self.client = genai.Client(api_key=self.settings.api_key.get_secret_value())
            self.model_name = self.settings.embedding_model_name
            logger.info(f'Initialized GeminiEmbedding with model: {self.model_name}')
        except Exception as e:
            raise EmbeddingConfigurationError(
                f'Failed to initialize Gemini embedding client: {str(e)}',
                details={'provider': 'gemini', 'error': str(e)}
            )

    @rate_limit(RateLimiterFactory.get_gemini_limiter)
    @retry(
        stop=stop_after_attempt(settings.gemini.max_retries), 
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((errors.ServerError,))
    )
    async def embed_text(self, text: str):
        """Generate embeddings for a single text."""
        try:
            if not text or not text.strip():
                logger.warning("Attempting to embed empty text")
                raise EmbeddingResponseError(
                    "Cannot embed empty or whitespace-only text",
                    provider='gemini',
                    details={'model': self.model_name}
                )

            logger.debug(f"Generating embedding for text of length: {len(text)}")
            response = await self.client.aio.models.embed_content(
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=settings.rag.embedding_dimension)
            )

            if not hasattr(response, 'embeddings') or not response.embeddings:
                logger.error("Gemini embeddings API returned no embeddings")
                raise EmbeddingResponseError(
                    "Gemini embeddings API returned no embeddings",
                    provider='gemini',
                    expected_count=1,
                    received_count=0,
                    details={'model': self.model_name}
                )

            # Gemini returns the embedding directly
            embedding = response.embeddings
            
            # Validate embedding is a list and has correct dimension
            if not isinstance(embedding, list):
                logger.error(f"Gemini returned non-list embedding: {type(embedding)}")
                raise EmbeddingResponseError(
                    f"Gemini returned invalid embedding type: {type(embedding)}",
                    provider='gemini',
                    response_data=str(embedding),
                    details={'model': self.model_name}
                )
            
            if len(embedding) != settings.rag.embedding_dimension:
                logger.error(f"Embedding dimension mismatch: expected {settings.rag.embedding_dimension}, got {len(embedding)}")
                raise EmbeddingDimensionMismatchError(
                    "Embedding dimension does not match expected dimension",
                    expected_dimension=settings.rag.embedding_dimension,
                    received_dimension=len(embedding),
                    details={'model': self.model_name}
                )

            logger.debug(f"Successfully generated embedding with dimension {len(embedding)}")
            return embedding

        except errors.ClientError as e:
            error_message = str(e).lower()
            if 'auth' in error_message or 'permission' in error_message or 'api key' in error_message:
                logger.error(f"Gemini authentication failed: {str(e)}")
                raise EmbeddingAuthenticationError(
                    f"Gemini authentication failed. Please verify your API key is correct and active.",
                    provider='gemini',
                    details={'error': str(e), 'model': self.model_name}
                ) from e
            if 'rate' in error_message or 'quota' in error_message or 'limit' in error_message:
                logger.warning(f"Gemini rate limit exceeded: {str(e)}")
                raise EmbeddingRateLimitError(
                    f"Gemini rate limit exceeded. Please wait before retrying.",
                    provider='gemini',
                    details={'error': str(e), 'model': self.model_name, 'max_retries': settings.gemini.max_retries}
                ) from e
            if 'timeout' in error_message or 'deadline' in error_message:
                logger.error(f"Gemini API timeout: {str(e)}")
                raise EmbeddingTimeoutError(
                    f"Gemini embeddings API request timed out",
                    provider='gemini',
                    details={'error': str(e), 'model': self.model_name}
                ) from e
            logger.error(f"Gemini client error: {str(e)}")
            raise EmbeddingAPIError(
                f"Gemini client error: {str(e)}",
                provider='gemini',
                model=self.model_name,
                text_count=1,
                error_type='client_error',
                details={'error': str(e)}
            ) from e
        except errors.ServerError as e:
            logger.error(f"Gemini server error: {str(e)}")
            raise EmbeddingAPIError(
                f"Gemini server error: {str(e)}",
                provider='gemini',
                model=self.model_name,
                text_count=1,
                error_type='server_error',
                details={'error': str(e)}
            ) from e
        except errors.APIError as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise EmbeddingAPIError(
                f"Gemini embeddings API error: {str(e)}",
                provider='gemini',
                model=self.model_name,
                text_count=1,
                error_type=type(e).__name__,
                details={'error': str(e)}
            ) from e
        
        except (EmbeddingResponseError, EmbeddingDimensionMismatchError):
            # Re-raise our custom exceptions
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error during Gemini embedding: {str(e)}", exc_info=True)
            raise EmbeddingAPIError(
                f"Unexpected error during Gemini embedding generation: {str(e)}",
                provider='gemini',
                model=self.model_name,
                text_count=1,
                error_type='unexpected_error',
                details={'error': str(e), 'error_type': type(e).__name__}
            ) from e

    # embed_texts is now handled by the base class


    @rate_limit(RateLimiterFactory.get_gemini_limiter)
    @retry(
        stop=stop_after_attempt(settings.gemini.max_retries), 
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((errors.ServerError,))
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a single batch (internal method with rate limiting and retry)."""
        logger.debug(f"Generating embeddings for batch of {len(texts)} texts")
        response = await self.client.aio.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=settings.rag.embedding_dimension)
        )

        if not hasattr(response, 'embeddings') or not response.embeddings or len(response.embeddings) == 0:
            logger.error('Gemini embeddings API returned no embeddings for batch')
            raise EmbeddingResponseError(
                'Gemini embeddings API returned no embeddings for batch',
                provider='gemini',
                expected_count=len(texts),
                received_count=0,
                details={'model': self.model_name}
            )
        
        if len(response.embeddings) != len(texts):
            logger.error(f"Embedding count mismatch: expected {len(texts)}, got {len(response.embeddings)}")
            raise EmbeddingResponseError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(response.embeddings)}",
                provider='gemini',
                expected_count=len(texts),
                received_count=len(response.embeddings),
                details={'model': self.model_name}
            )
        
        embeddings = []
        for i, embed in enumerate(response.embeddings):
            if not isinstance(embed, list):
                logger.error(f"Gemini returned non-list embedding at index {i}: {type(embed)}")
                raise EmbeddingResponseError(
                    f"Gemini returned invalid embedding type at index {i}: {type(embed)}",
                    provider='gemini',
                    response_data=str(embed),
                    details={'model': self.model_name, 'index': i}
                )
            
            if len(embed) != settings.rag.embedding_dimension:
                logger.error(f"Embedding {i} dimension mismatch: expected {settings.rag.embedding_dimension}, got {len(embed)}")
                raise EmbeddingDimensionMismatchError(
                    f"Embedding dimension does not match expected dimension at index {i}",
                    expected_dimension=settings.rag.embedding_dimension,
                    received_dimension=len(embed),
                    details={'model': self.model_name, 'index': i}
                )
            
            embeddings.append(embed)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return settings.rag.embedding_dimension
