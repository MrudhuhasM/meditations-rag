"""Core modules for the Meditations RAG application."""

from meditations_rag.core.exceptions import (
    MeditationsRAGException,
    # LLM Exceptions
    LLMException,
    LLMConfigurationError,
    LLMAPIError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMResponseError,
    LLMAuthenticationError,
    # Embedding Exceptions
    EmbeddingException,
    EmbeddingConfigurationError,
    EmbeddingAPIError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    EmbeddingResponseError,
    EmbeddingAuthenticationError,
    EmbeddingDimensionMismatchError,
    # Vector Store Exceptions
    VectorStoreException,
    VectorStoreConnectionError,
    VectorStoreQueryError,
)

__all__ = [
    "MeditationsRAGException",
    # LLM Exceptions
    "LLMException",
    "LLMConfigurationError",
    "LLMAPIError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMResponseError",
    "LLMAuthenticationError",
    # Embedding Exceptions
    "EmbeddingException",
    "EmbeddingConfigurationError",
    "EmbeddingAPIError",
    "EmbeddingRateLimitError",
    "EmbeddingTimeoutError",
    "EmbeddingResponseError",
    "EmbeddingAuthenticationError",
    "EmbeddingDimensionMismatchError",
    # Vector Store Exceptions
    "VectorStoreException",
    "VectorStoreConnectionError",
    "VectorStoreQueryError",
]
