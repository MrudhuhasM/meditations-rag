"""
Custom exceptions for the Meditations RAG application.

These exceptions provide clear, actionable error messages for debugging
LLM, embedding, and vector store failures.
"""

from typing import Optional, Any


class MeditationsRAGException(Exception):
    """Base exception for all Meditations RAG errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} | Details: {details_str}"
        return self.message


# ============================================================================
# LLM Exceptions
# ============================================================================

class LLMException(MeditationsRAGException):
    """Base exception for LLM-related errors."""
    pass


class LLMConfigurationError(LLMException):
    """Raised when LLM configuration is invalid or missing."""
    pass


class LLMAPIError(LLMException):
    """Raised when LLM API calls fail."""
    
    def __init__(
        self, 
        message: str, 
        provider: str,
        model: str,
        status_code: Optional[int] = None,
        error_type: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "provider": provider,
            "model": model,
            "status_code": status_code,
            "error_type": error_type,
        })
        super().__init__(message, details)


class LLMRateLimitError(LLMException):
    """Raised when LLM rate limits are exceeded."""
    
    def __init__(
        self, 
        message: str, 
        provider: str,
        retry_after: Optional[int] = None,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "provider": provider,
            "retry_after": retry_after,
        })
        super().__init__(message, details)


class LLMTimeoutError(LLMException):
    """Raised when LLM API calls timeout."""
    
    def __init__(
        self, 
        message: str, 
        provider: str,
        timeout_seconds: Optional[float] = None,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "provider": provider,
            "timeout_seconds": timeout_seconds,
        })
        super().__init__(message, details)


class LLMResponseError(LLMException):
    """Raised when LLM returns invalid or empty responses."""
    
    def __init__(
        self, 
        message: str, 
        provider: str,
        response_data: Optional[Any] = None,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "provider": provider,
            "response_data": str(response_data) if response_data else None,
        })
        super().__init__(message, details)


class LLMAuthenticationError(LLMException):
    """Raised when LLM authentication fails."""
    
    def __init__(
        self, 
        message: str, 
        provider: str,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        details.update({"provider": provider})
        super().__init__(message, details)


# ============================================================================
# Embedding Exceptions
# ============================================================================

class EmbeddingException(MeditationsRAGException):
    """Base exception for embedding-related errors."""
    pass


class EmbeddingConfigurationError(EmbeddingException):
    """Raised when embedding configuration is invalid or missing."""
    pass


class EmbeddingAPIError(EmbeddingException):
    """Raised when embedding API calls fail."""
    
    def __init__(
        self, 
        message: str, 
        provider: str,
        model: str,
        text_count: Optional[int] = None,
        status_code: Optional[int] = None,
        error_type: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "provider": provider,
            "model": model,
            "text_count": text_count,
            "status_code": status_code,
            "error_type": error_type,
        })
        super().__init__(message, details)


class EmbeddingRateLimitError(EmbeddingException):
    """Raised when embedding rate limits are exceeded."""
    
    def __init__(
        self, 
        message: str, 
        provider: str,
        retry_after: Optional[int] = None,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "provider": provider,
            "retry_after": retry_after,
        })
        super().__init__(message, details)


class EmbeddingTimeoutError(EmbeddingException):
    """Raised when embedding API calls timeout."""
    
    def __init__(
        self, 
        message: str, 
        provider: str,
        timeout_seconds: Optional[float] = None,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "provider": provider,
            "timeout_seconds": timeout_seconds,
        })
        super().__init__(message, details)


class EmbeddingResponseError(EmbeddingException):
    """Raised when embedding returns invalid or empty responses."""
    
    def __init__(
        self, 
        message: str, 
        provider: str,
        expected_count: Optional[int] = None,
        received_count: Optional[int] = None,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "provider": provider,
            "expected_count": expected_count,
            "received_count": received_count,
        })
        super().__init__(message, details)


class EmbeddingAuthenticationError(EmbeddingException):
    """Raised when embedding authentication fails."""
    
    def __init__(
        self, 
        message: str, 
        provider: str,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        details.update({"provider": provider})
        super().__init__(message, details)


class EmbeddingDimensionMismatchError(EmbeddingException):
    """Raised when embedding dimensions don't match expected values."""
    
    def __init__(
        self, 
        message: str, 
        expected_dimension: int,
        received_dimension: int,
        details: Optional[dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "expected_dimension": expected_dimension,
            "received_dimension": received_dimension,
        })
        super().__init__(message, details)


# ============================================================================
# Vector Store Exceptions
# ============================================================================

class VectorStoreException(MeditationsRAGException):
    """Base exception for vector store errors."""
    pass


class VectorStoreConnectionError(VectorStoreException):
    """Raised when vector store connection fails."""
    pass


class VectorStoreQueryError(VectorStoreException):
    """Raised when vector store queries fail."""
    pass
