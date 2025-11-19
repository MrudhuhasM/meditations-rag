from .base import EmbeddingBase
from .openai import OpenAIEmbedding
from .gemini import GeminiEmbedding
from .local_embedding import LocalEmbedding
from meditations_rag.config import settings, get_logger


def create_embedding(provider: str | None = None) -> EmbeddingBase:
    """
    Factory function to create embedding instances based on provider.
    
    Uses Factory pattern for flexible provider selection.
    
    Args:
        provider: Embedding provider name ('openai', 'gemini', 'local').
                 If None, uses settings.rag.embedding_provider
    
    Returns:
        EmbeddingBase implementation for the specified provider
        
    Raises:
        ValueError: If provider is not supported
    """
    provider = provider or settings.rag.embedding_provider
    provider = provider.lower()
    
    if provider == 'openai':
        return OpenAIEmbedding()
    elif provider == 'gemini':
        return GeminiEmbedding()
    elif provider == 'local':
        return LocalEmbedding()
    else:
        raise ValueError(f'Unsupported embedding provider: {provider}')


__all__ = ['EmbeddingBase', 'OpenAIEmbedding', 'GeminiEmbedding', 'LocalEmbedding', 'create_embedding']
