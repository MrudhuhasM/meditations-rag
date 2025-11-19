from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError, AuthenticationError, APIConnectionError
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


class OpenAIEmbedding(EmbeddingBase):
    """OpenAI embedding implementation."""
    
    def __init__(self):
        self.settings = settings.openai

        if not self.settings.api_key:
            raise EmbeddingConfigurationError(
                'OpenAI API key is not configured. Please set OPENAI_API_KEY in your environment.',
                details={'provider': 'openai'}
            )

        try:
            self.client = AsyncOpenAI(api_key=self.settings.api_key.get_secret_value())
            self.model_name = self.settings.embedding_model_name
            logger.info(f'Initialized OpenAIEmbedding with model: {self.model_name}')
        except Exception as e:
            raise EmbeddingConfigurationError(
                f'Failed to initialize OpenAI embedding client: {str(e)}',
                details={'provider': 'openai', 'error': str(e)}
            )

    @rate_limit(RateLimiterFactory.get_openai_limiter)
    @retry(
        stop=stop_after_attempt(settings.openai.max_retries), 
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError))
    )
    async def embed_text(self, text: str) -> list[float]:
        """Generate embeddings for a single text."""
        try:
            if not text or not text.strip():
                logger.warning("Attempting to embed empty text")
                raise EmbeddingResponseError(
                    "Cannot embed empty or whitespace-only text",
                    provider='openai',
                    details={'model': self.model_name}
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
                    provider='openai',
                    expected_count=1,
                    received_count=0,
                    details={'model': self.model_name}
                )

            embedding = response.data[0].embedding
            
            # Validate embedding dimension
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

        except AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {str(e)}")
            raise EmbeddingAuthenticationError(
                f"OpenAI authentication failed. Please verify your API key is correct and active.",
                provider='openai',
                details={'error': str(e), 'model': self.model_name}
            ) from e
        
        except RateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded: {str(e)}")
            raise EmbeddingRateLimitError(
                f"OpenAI rate limit exceeded. Please wait before retrying.",
                provider='openai',
                details={
                    'error': str(e),
                    'model': self.model_name,
                    'max_retries': settings.openai.max_retries
                }
            ) from e
        
        except APITimeoutError as e:
            logger.error(f"OpenAI API timeout: {str(e)}")
            raise EmbeddingTimeoutError(
                f"OpenAI embeddings API request timed out",
                provider='openai',
                details={
                    'error': str(e),
                    'model': self.model_name
                }
            ) from e
        
        except APIConnectionError as e:
            logger.error(f"OpenAI API connection error: {str(e)}")
            raise EmbeddingAPIError(
                f"Failed to connect to OpenAI embeddings API. Check your network connection.",
                provider='openai',
                model=self.model_name,
                text_count=1,
                error_type='connection_error',
                details={'error': str(e)}
            ) from e
        
        except APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise EmbeddingAPIError(
                f"OpenAI embeddings API error: {str(e)}",
                provider='openai',
                model=self.model_name,
                text_count=1,
                status_code=getattr(e, 'status_code', None),
                error_type=getattr(e, 'type', None),
                details={'error': str(e)}
            ) from e
        
        except (EmbeddingResponseError, EmbeddingDimensionMismatchError):
            # Re-raise our custom exceptions
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI embedding: {str(e)}", exc_info=True)
            raise EmbeddingAPIError(
                f"Unexpected error during OpenAI embedding generation: {str(e)}",
                provider='openai',
                model=self.model_name,
                text_count=1,
                error_type='unexpected_error',
                details={'error': str(e), 'error_type': type(e).__name__}
            ) from e

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch."""
        try:
            if not texts or len(texts) == 0:
                logger.warning("Attempting to embed empty text list")
                raise EmbeddingResponseError(
                    "Cannot embed empty list of texts",
                    provider='openai',
                    expected_count=0,
                    received_count=0,
                    details={'model': self.model_name}
                )

            # Filter out empty texts and log warning
            valid_texts = [t for t in texts if t and t.strip()]
            if len(valid_texts) < len(texts):
                logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts from batch")
                if len(valid_texts) == 0:
                    raise EmbeddingResponseError(
                        "All texts in batch are empty or whitespace-only",
                        provider='openai',
                        expected_count=len(texts),
                        received_count=0,
                        details={'model': self.model_name}
                    )

            # Split into batches if needed (API limit is typically 32)
            api_batch_size = settings.rag.embedding_api_batch_size
            all_embeddings = []
            
            if len(valid_texts) <= api_batch_size:
                # Single batch - process directly
                logger.debug(f"Generating embeddings for {len(valid_texts)} texts in single batch")
                batch_embeddings = await self._embed_batch(valid_texts)
                all_embeddings.extend(batch_embeddings)
            else:
                # Multiple batches needed
                num_batches = (len(valid_texts) + api_batch_size - 1) // api_batch_size
                logger.info(f"Splitting {len(valid_texts)} texts into {num_batches} batches of max {api_batch_size}")
                
                for i in range(0, len(valid_texts), api_batch_size):
                    batch = valid_texts[i:i + api_batch_size]
                    batch_num = (i // api_batch_size) + 1
                    logger.debug(f"Processing batch {batch_num}/{num_batches} with {len(batch)} texts")
                    
                    batch_embeddings = await self._embed_batch(batch)
                    all_embeddings.extend(batch_embeddings)
            
            logger.debug(f"Successfully generated {len(all_embeddings)} embeddings")
            return all_embeddings

        except AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {str(e)}")
            raise EmbeddingAuthenticationError(
                f"OpenAI authentication failed. Please verify your API key is correct and active.",
                provider='openai',
                details={'error': str(e), 'model': self.model_name}
            ) from e
        
        except RateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded: {str(e)}")
            raise EmbeddingRateLimitError(
                f"OpenAI rate limit exceeded. Please wait before retrying.",
                provider='openai',
                details={
                    'error': str(e),
                    'model': self.model_name,
                    'max_retries': settings.openai.max_retries,
                    'text_count': len(texts)
                }
            ) from e
        
        except APITimeoutError as e:
            logger.error(f"OpenAI API timeout: {str(e)}")
            raise EmbeddingTimeoutError(
                f"OpenAI embeddings API request timed out",
                provider='openai',
                details={
                    'error': str(e),
                    'model': self.model_name,
                    'text_count': len(texts)
                }
            ) from e
        
        except APIConnectionError as e:
            logger.error(f"OpenAI API connection error: {str(e)}")
            raise EmbeddingAPIError(
                f"Failed to connect to OpenAI embeddings API. Check your network connection.",
                provider='openai',
                model=self.model_name,
                text_count=len(texts),
                error_type='connection_error',
                details={'error': str(e)}
            ) from e
        
        except APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise EmbeddingAPIError(
                f"OpenAI embeddings API error: {str(e)}",
                provider='openai',
                model=self.model_name,
                text_count=len(texts),
                status_code=getattr(e, 'status_code', None),
                error_type=getattr(e, 'type', None),
                details={'error': str(e)}
            ) from e
        
        except (EmbeddingResponseError, EmbeddingDimensionMismatchError):
            # Re-raise our custom exceptions
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI batch embedding: {str(e)}", exc_info=True)
            raise EmbeddingAPIError(
                f"Unexpected error during OpenAI batch embedding generation: {str(e)}",
                provider='openai',
                model=self.model_name,
                text_count=len(texts),
                error_type='unexpected_error',
                details={'error': str(e), 'error_type': type(e).__name__}
            ) from e

    @rate_limit(RateLimiterFactory.get_openai_limiter)
    @retry(
        stop=stop_after_attempt(settings.openai.max_retries), 
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError))
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
                provider='openai',
                expected_count=len(texts),
                received_count=0,
                details={'model': self.model_name})

        if len(response.data) != len(texts):
            logger.error(f"Embedding count mismatch: expected {len(texts)}, got {len(response.data)}")
            raise EmbeddingResponseError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(response.data)}",
                provider='openai',
                expected_count=len(texts),
                received_count=len(response.data),
                details={'model': self.model_name}
            )
        
        # Ensure results are in the same order as input and validate dimensions
        embeddings = []
        for i, item in enumerate(response.data):
            if len(item.embedding) != settings.rag.embedding_dimension:
                logger.error(f"Embedding {i} dimension mismatch: expected {settings.rag.embedding_dimension}, got {len(item.embedding)}")
                raise EmbeddingDimensionMismatchError(
                    f"Embedding dimension does not match expected dimension at index {i}",
                    expected_dimension=settings.rag.embedding_dimension,
                    received_dimension=len(item.embedding),
                    details={'model': self.model_name, 'index': i}
                )
            embeddings.append(item.embedding)

        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return settings.rag.embedding_dimension
