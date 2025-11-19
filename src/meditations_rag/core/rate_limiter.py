"""
Rate limiting utilities for API calls using PyrateLimiter.

This module provides rate limiters for different API providers to prevent
hitting provider rate limits. It uses the leaky bucket algorithm for precise
control over request rates.
"""

from functools import wraps
from typing import Callable, TypeVar
from collections.abc import Coroutine
from pyrate_limiter import Duration, Rate, Limiter, BucketFullException, InMemoryBucket
from meditations_rag.config import settings, get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class RateLimiterFactory:
    """Factory for creating rate limiters for different providers."""
    
    _openai_limiter: Limiter | None = None
    _gemini_limiter: Limiter | None = None
    _openrouter_limiter: Limiter | None = None
    
    @classmethod
    def get_openai_limiter(cls) -> Limiter | None:
        """
        Get or create the OpenAI rate limiter.
        
        Returns:
            Limiter instance if rate limiting is enabled, None otherwise
        """
        if not settings.openai.rate_limit_enabled:
            return None
            
        if cls._openai_limiter is None:
            # OpenAI has per-minute limits for both requests and tokens
            # We focus on requests per minute here
            rates = [
                Rate(settings.openai.rate_limit_requests_per_minute, Duration.MINUTE)
            ]
            bucket = InMemoryBucket(rates)
            cls._openai_limiter = Limiter(bucket)
            logger.info(
                f"Initialized OpenAI rate limiter: "
                f"{settings.openai.rate_limit_requests_per_minute} requests/minute"
            )
        
        return cls._openai_limiter
    
    @classmethod
    def get_gemini_limiter(cls) -> Limiter | None:
        """
        Get or create the Gemini rate limiter.
        
        Returns:
            Limiter instance if rate limiting is enabled, None otherwise
        """
        if not settings.gemini.rate_limit_enabled:
            return None
            
        if cls._gemini_limiter is None:
            # Gemini has per-minute and per-day limits
            rates = [
                Rate(settings.gemini.rate_limit_requests_per_minute, Duration.MINUTE),
                Rate(settings.gemini.rate_limit_requests_per_day, Duration.DAY)
            ]
            bucket = InMemoryBucket(rates)
            cls._gemini_limiter = Limiter(bucket)
            logger.info(
                f"Initialized Gemini rate limiter: "
                f"{settings.gemini.rate_limit_requests_per_minute} requests/minute, "
                f"{settings.gemini.rate_limit_requests_per_day} requests/day"
            )
        
        return cls._gemini_limiter
    
    @classmethod
    def get_openrouter_limiter(cls) -> Limiter | None:
        """
        Get or create the OpenRouter rate limiter.
        
        Returns:
            Limiter instance if rate limiting is enabled, None otherwise
        """
        if not settings.openrouter.rate_limit_enabled:
            return None
            
        if cls._openrouter_limiter is None:
            # OpenRouter has per-minute limits for requests
            rates = [
                Rate(settings.openrouter.rate_limit_requests_per_minute, Duration.MINUTE)
            ]
            bucket = InMemoryBucket(rates)
            cls._openrouter_limiter = Limiter(bucket)
            logger.info(
                f"Initialized OpenRouter rate limiter: "
                f"{settings.openrouter.rate_limit_requests_per_minute} requests/minute"
            )
        
        return cls._openrouter_limiter


def rate_limit(limiter_getter: Callable[[], Limiter | None]):
    """
    Decorator to apply rate limiting to async functions.
    
    Args:
        limiter_getter: Function that returns the appropriate Limiter instance
        
    Example:
        @rate_limit(RateLimiterFactory.get_openai_limiter)
        async def my_api_call():
            ...
    """
    def decorator(func: Callable[..., Coroutine[object, object, T]]) -> Callable[..., Coroutine[object, object, T]]:
        @wraps(func)
        async def async_wrapper(*args: object, **kwargs: object) -> T:
            limiter = limiter_getter()
            
            if limiter is None:
                # Rate limiting disabled, call function directly
                return await func(*args, **kwargs)
            
            try:
                # Acquire rate limit slot
                limiter.try_acquire(func.__name__)
            except BucketFullException as e:
                logger.warning(
                    f"Rate limit exceeded for {func.__name__}: {e}. "
                    f"Waiting before retry..."
                )
                raise
            
            return await func(*args, **kwargs)
        
        return async_wrapper
    
    return decorator
