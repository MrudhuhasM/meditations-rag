"""
Chunk Embedding Module for Meditations RAG.

This module provides embedding models specifically for semantic chunking operations.
These embeddings are used by SemanticSplitterNodeParser to intelligently chunk documents.

Design Pattern: Factory + Strategy Pattern
- Factory: ChunkEmbeddingFactory creates appropriate embedding instances
- Strategy: Different embedding providers implement the same interface

Supported Providers:
    - OpenAI: text-embedding-ada-002 and text-embedding-3-* models
    - Gemini: Google's embedding models via GenAI API
    - Local LLM: OpenAI-compatible local models (e.g., Ollama, LM Studio)

Note: This module uses LlamaIndex's embedding abstractions. In the future,
      a separate module will provide raw API-based embeddings for vector storage.
"""

from abc import ABC, abstractmethod
from typing import Optional
from functools import lru_cache

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from pydantic import SecretStr

from meditations_rag.config.settings import settings
from meditations_rag.config.logger import get_logger

logger = get_logger(__name__)


class EmbeddingProviderError(Exception):
    """Raised when there's an error with the embedding provider."""
    pass


class EmbeddingConfigurationError(Exception):
    """Raised when embedding configuration is invalid or missing."""
    pass


class IChunkEmbeddingProvider(ABC):
    """
    Interface for chunk embedding providers.
    
    This defines the contract that all embedding providers must implement.
    Following Interface Segregation Principle (ISP) - clients depend only on this interface.
    """
    
    @abstractmethod
    def get_embedding_model(self) -> BaseEmbedding:
        """
        Get the configured embedding model instance.
        
        Returns:
            BaseEmbedding: LlamaIndex embedding model instance
            
        Raises:
            EmbeddingProviderError: If model creation fails
            EmbeddingConfigurationError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def validate_configuration(self) -> bool:
        """
        Validate that all required configuration is present.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            EmbeddingConfigurationError: If configuration is invalid or missing
        """
        pass


class OpenAIChunkEmbeddingProvider(IChunkEmbeddingProvider):
    """
    OpenAI embedding provider for semantic chunking.
    
    Supports OpenAI's embedding models via official API.
    Default model: text-embedding-ada-002 (1536 dimensions)
    
    Configuration via settings:
        - OPENAI_API_KEY: API key (required)
        - OPENAI_API_BASE: API base URL (optional)
        - OPENAI_EMBEDDING_MODEL_NAME: Model name (optional)
        - RAG_CHUNK_EMBED_BATCH_SIZE: Batch size (optional)
    """
    
    def __init__(self):
        """Initialize OpenAI provider with settings."""
        self._settings = settings.openai
        self._rag_settings = settings.rag
        
    def validate_configuration(self) -> bool:
        """Validate OpenAI configuration."""
        if self._settings.api_key is None:
            raise EmbeddingConfigurationError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )
        
        if not self._settings.api_key.get_secret_value().strip():
            raise EmbeddingConfigurationError(
                "OpenAI API key cannot be empty."
            )
        
        logger.info(
            "OpenAI embedding configuration validated",
            extra={
                "model": self._settings.embedding_model_name,
                "api_base": self._settings.api_base,
                "batch_size": self._rag_settings.chunk_embed_batch_size
            }
        )
        return True
    
    def get_embedding_model(self) -> BaseEmbedding:
        """Create OpenAI embedding model instance."""
        self.validate_configuration()
        
        # Type assertion: validation ensures api_key is not None
        assert self._settings.api_key is not None, "API key must be set"
        
        try:
            model = OpenAIEmbedding(
                model=self._settings.embedding_model_name,
                api_key=self._settings.api_key.get_secret_value(),
                api_base=self._settings.api_base,
                embed_batch_size=self._rag_settings.chunk_embed_batch_size,
                timeout=self._settings.timeout,
                max_retries=self._settings.max_retries,
            )
            
            logger.info(
                "OpenAI chunk embedding model created successfully",
                extra={"model": self._settings.embedding_model_name}
            )
            return model
            
        except Exception as e:
            logger.error(
                "Failed to create OpenAI embedding model",
                extra={"error": str(e)},
                exc_info=True
            )
            raise EmbeddingProviderError(
                f"Failed to initialize OpenAI embedding model: {str(e)}"
            ) from e


class GeminiChunkEmbeddingProvider(IChunkEmbeddingProvider):
    """
    Google Gemini embedding provider for semantic chunking.
    
    Supports Google's Gemini embedding models via GenAI API.
    Default model: models/text-embedding-004 (768 dimensions)
    
    Configuration via settings:
        - GEMINI_API_KEY: API key (required)
        - GEMINI_EMBEDDING_MODEL_NAME: Model name (optional)
        - RAG_CHUNK_EMBED_BATCH_SIZE: Batch size (optional)
    """
    
    def __init__(self):
        """Initialize Gemini provider with settings."""
        self._settings = settings.gemini
        self._rag_settings = settings.rag
        
    def validate_configuration(self) -> bool:
        """Validate Gemini configuration."""
        if self._settings.api_key is None:
            raise EmbeddingConfigurationError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable."
            )
        
        if not self._settings.api_key.get_secret_value().strip():
            raise EmbeddingConfigurationError(
                "Gemini API key cannot be empty."
            )
        
        logger.info(
            "Gemini embedding configuration validated",
            extra={
                "model": self._settings.embedding_model_name,
                "batch_size": self._rag_settings.chunk_embed_batch_size
            }
        )
        return True
    
    def get_embedding_model(self) -> BaseEmbedding:
        """Create Gemini embedding model instance."""
        self.validate_configuration()
        
        # Type assertion: validation ensures api_key is not None
        assert self._settings.api_key is not None, "API key must be set"
        
        try:
            # Use GoogleGenAIEmbedding (newer API)
            model = GoogleGenAIEmbedding(
                model_name=self._settings.embedding_model_name,
                api_key=self._settings.api_key.get_secret_value(),
                # GoogleGenAIEmbedding doesn't support embed_batch_size parameter
                # Batching is handled internally
            )
            
            logger.info(
                "Gemini chunk embedding model created successfully",
                extra={"model": self._settings.embedding_model_name}
            )
            return model
            
        except Exception as e:
            logger.error(
                "Failed to create Gemini embedding model",
                extra={"error": str(e)},
                exc_info=True
            )
            raise EmbeddingProviderError(
                f"Failed to initialize Gemini embedding model: {str(e)}"
            ) from e


class LocalLLMChunkEmbeddingProvider(IChunkEmbeddingProvider):
    """
    Local LLM embedding provider for semantic chunking.
    
    Supports OpenAI-compatible local embedding services (e.g., Ollama, LM Studio).
    Uses OpenAI client with custom base URL.
    
    Configuration via settings:
        - LOCAL_LLM_API_EMBEDDING_BASE_URL: Base URL for embedding service (required)
        - LOCAL_LLM_EMBEDDING_MODEL_NAME: Model name (required)
        - LOCAL_LLM_API_KEY: API key if required by service (optional)
        - RAG_CHUNK_EMBED_BATCH_SIZE: Batch size (optional)
    """
    
    def __init__(self):
        """Initialize Local LLM provider with settings."""
        self._settings = settings.local_llm
        self._rag_settings = settings.rag
        
    def validate_configuration(self) -> bool:
        """Validate Local LLM configuration."""
        if not self._settings.api_embedding_base_url:
            raise EmbeddingConfigurationError(
                "Local LLM embedding base URL is required. "
                "Set LOCAL_LLM_API_EMBEDDING_BASE_URL environment variable."
            )
        
        if not self._settings.embedding_model_name:
            raise EmbeddingConfigurationError(
                "Local LLM embedding model name is required. "
                "SET LOCAL_LLM_EMBEDDING_MODEL_NAME environment variable."
            )
        
        logger.info(
            "Local LLM embedding configuration validated",
            extra={
                "model": self._settings.embedding_model_name,
                "api_base": self._settings.api_embedding_base_url,
                "batch_size": self._rag_settings.chunk_embed_batch_size
            }
        )
        return True
    
    def get_embedding_model(self) -> BaseEmbedding:
        """Create Local LLM embedding model instance using OpenAI client."""
        self.validate_configuration()
        
        # Type assertions: validation ensures these are not None
        assert self._settings.embedding_model_name is not None, "Model name must be set"
        assert self._settings.api_embedding_base_url is not None, "API base URL must be set"
        
        try:
            # Use OpenAI client with custom base URL for local models
            # This works with any OpenAI-compatible API (Ollama, LM Studio, etc.)
            api_key_value = (
                self._settings.api_key.get_secret_value() 
                if self._settings.api_key 
                else "not-needed"  # Some local services don't require API key
            )
            
            model = OpenAIEmbedding(
                model_name=self._settings.embedding_model_name,
                api_key=api_key_value,
                api_base=self._settings.api_embedding_base_url,
                embed_batch_size=self._rag_settings.chunk_embed_batch_size,
                timeout=self._settings.timeout,
                max_retries=self._settings.max_retries,
            )
            
            logger.info(
                "Local LLM chunk embedding model created successfully",
                extra={
                    "model": self._settings.embedding_model_name,
                    "api_base": self._settings.api_embedding_base_url
                }
            )
            return model
            
        except Exception as e:
            logger.error(
                "Failed to create Local LLM embedding model",
                extra={"error": str(e)},
                exc_info=True
            )
            raise EmbeddingProviderError(
                f"Failed to initialize Local LLM embedding model: {str(e)}"
            ) from e


class ChunkEmbeddingFactory:
    """
    Factory for creating chunk embedding model instances.
    
    Design Pattern: Factory Pattern + Singleton (via lru_cache)
    
    This factory creates the appropriate embedding provider based on configuration.
    It follows the Open/Closed Principle - new providers can be added without
    modifying existing code.
    
    Usage:
        factory = ChunkEmbeddingFactory()
        embedding_model = factory.create_embedding_model()
    
    The provider is selected based on RAG_EMBEDDING_PROVIDER setting:
        - "openai": OpenAI embeddings
        - "gemini": Google Gemini embeddings
        - "local": Local LLM embeddings (OpenAI-compatible)
    """
    
    # Provider registry - follows Open/Closed Principle
    _providers = {
        "openai": OpenAIChunkEmbeddingProvider,
        "gemini": GeminiChunkEmbeddingProvider,
        "local": LocalLLMChunkEmbeddingProvider,
    }
    
    def __init__(self):
        """Initialize the factory."""
        self._rag_settings = settings.rag
        
    def create_embedding_model(
        self, 
        provider: Optional[str] = None
    ) -> BaseEmbedding:
        """
        Create an embedding model instance.
        
        Args:
            provider: Override provider selection. If None, uses RAG_EMBEDDING_PROVIDER
                     setting. Supported values: "openai", "gemini", "local"
        
        Returns:
            BaseEmbedding: Configured embedding model instance
            
        Raises:
            EmbeddingConfigurationError: If provider is unknown or configuration invalid
            EmbeddingProviderError: If model creation fails
        """
        selected_provider = provider or self._rag_settings.embedding_provider
        selected_provider = selected_provider.lower()
        
        if selected_provider not in self._providers:
            available = ", ".join(self._providers.keys())
            raise EmbeddingConfigurationError(
                f"Unknown embedding provider: '{selected_provider}'. "
                f"Available providers: {available}"
            )
        
        logger.info(
            "Creating chunk embedding model",
            extra={"provider": selected_provider}
        )
        
        provider_class = self._providers[selected_provider]
        provider_instance = provider_class()
        
        return provider_instance.get_embedding_model()
    
    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: type[IChunkEmbeddingProvider]
    ) -> None:
        """
        Register a new embedding provider.
        
        This allows extending the factory with custom providers without
        modifying the core code (Open/Closed Principle).
        
        Args:
            name: Provider identifier (e.g., "custom_provider")
            provider_class: Class implementing IChunkEmbeddingProvider
            
        Example:
            ChunkEmbeddingFactory.register_provider(
                "custom",
                CustomEmbeddingProvider
            )
        """
        if not issubclass(provider_class, IChunkEmbeddingProvider):
            raise ValueError(
                f"Provider class must implement IChunkEmbeddingProvider interface"
            )
        
        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered new chunk embedding provider: {name}")


@lru_cache(maxsize=1)
def get_chunk_embedding_model(provider: Optional[str] = None) -> BaseEmbedding:
    """
    Get a cached chunk embedding model instance.
    
    This is a convenience function that creates a singleton embedding model.
    The model is cached to avoid recreating it on every call, which improves
    performance and reduces API overhead.
    
    Args:
        provider: Override provider selection. If None, uses RAG_EMBEDDING_PROVIDER
                 setting. Supported values: "openai", "gemini", "local"
    
    Returns:
        BaseEmbedding: Configured and cached embedding model instance
        
    Raises:
        EmbeddingConfigurationError: If provider is unknown or configuration invalid
        EmbeddingProviderError: If model creation fails
        
    Example:
        from meditations_rag.core.chunk_embedding import get_chunk_embedding_model
        
        # Use default provider from settings
        embed_model = get_chunk_embedding_model()
        
        # Override provider
        embed_model = get_chunk_embedding_model(provider="gemini")
    """
    factory = ChunkEmbeddingFactory()
    return factory.create_embedding_model(provider=provider)


__all__ = [
    "ChunkEmbeddingFactory",
    "get_chunk_embedding_model",
    "IChunkEmbeddingProvider",
    "OpenAIChunkEmbeddingProvider",
    "GeminiChunkEmbeddingProvider",
    "LocalLLMChunkEmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingConfigurationError",
]
