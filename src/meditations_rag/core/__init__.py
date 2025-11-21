"""Core modules for the Meditations RAG application."""

from meditations_rag.core.exceptions import (  # LLM Exceptions; Embedding Exceptions; Vector Store Exceptions
    EmbeddingAPIError,
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingDimensionMismatchError,
    EmbeddingException,
    EmbeddingRateLimitError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMException,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    MeditationsRAGException,
    VectorStoreConnectionError,
    VectorStoreException,
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
