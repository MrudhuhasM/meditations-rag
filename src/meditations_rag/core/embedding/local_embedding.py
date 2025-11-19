from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from meditations_rag.config import settings, get_logger
from meditations_rag.core.embedding.base import EmbeddingBase

logger = get_logger(__name__)


class LocalEmbedding(EmbeddingBase):
    """Local embedding implementation using OpenAI-compatible API."""
    
    def __init__(self):
        self.settings = settings.local_llm

        if not self.settings.api_embedding_base_url:
            raise ValueError('Local embedding API base URL is not configured.')

        self.client = AsyncOpenAI(
            api_key=self.settings.api_key.get_secret_value() if self.settings.api_key else 'not-needed',
            base_url=self.settings.api_embedding_base_url,
        )
        self.model_name = self.settings.embedding_model_name or 'local-embedding-model'
        logger.info(f'Initialized LocalEmbedding with model: {self.model_name}')

    @retry(stop=stop_after_attempt(settings.local_llm.max_retries), wait=wait_exponential())
    async def embed_text(self, text: str) -> list[float]:
        """Generate embeddings for a single text."""
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=text,
            dimensions=settings.rag.embedding_dimension,
        )
        
        embedding = response.data[0].embedding            
        return embedding

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch."""
        if not texts:
            return []
        
        # Split into batches if needed (API limit is typically 32)
        api_batch_size = settings.rag.embedding_api_batch_size
        all_embeddings = []
        
        if len(texts) <= api_batch_size:
            # Single batch - process directly
            logger.debug(f"Generating embeddings for {len(texts)} texts in single batch")
            batch_embeddings = await self._embed_batch(texts)
            all_embeddings.extend(batch_embeddings)
        else:
            # Multiple batches needed
            num_batches = (len(texts) + api_batch_size - 1) // api_batch_size
            logger.info(f"Splitting {len(texts)} texts into {num_batches} batches of max {api_batch_size}")
            
            for i in range(0, len(texts), api_batch_size):
                batch = texts[i:i + api_batch_size]
                batch_num = (i // api_batch_size) + 1
                logger.debug(f"Processing batch {batch_num}/{num_batches} with {len(batch)} texts")
                
                batch_embeddings = await self._embed_batch(batch)
                all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    @retry(stop=stop_after_attempt(settings.local_llm.max_retries), wait=wait_exponential())
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a single batch (internal method with retry)."""
        logger.debug(f"Generating embeddings for batch of {len(texts)} texts")
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            dimensions=settings.rag.embedding_dimension,
        )
        
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return settings.rag.embedding_dimension
