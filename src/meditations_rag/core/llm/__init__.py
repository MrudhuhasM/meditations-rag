from meditations_rag.config import settings

from .base import LLMBase
from .gemini import GeminiLLM
from .local_llm import LocalLLM
from .openai import OpenAILLM
from .openrouter import OpenRouterLLM


def create_llm(provider: str | None = None) -> LLMBase:
    """
    Factory function to create LLM instances based on provider.

    Uses Factory pattern for flexible provider selection.

    Args:
        provider: LLM provider name ('openai', 'gemini', 'local', 'openrouter').
                 If None, uses settings.rag.llm_provider

    Returns:
        LLMBase implementation for the specified provider

    Raises:
        ValueError: If provider is not supported
    """
    provider = provider or settings.rag.llm_provider
    provider = provider.lower()

    if provider == "openai":
        return OpenAILLM()
    elif provider == "gemini":
        return GeminiLLM()
    elif provider == "local":
        return LocalLLM()
    elif provider == "openrouter":
        return OpenRouterLLM()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def create_strong_llm(provider: str | None = None) -> LLMBase:
    """
    Factory function to create strong LLM instances based on provider.

    Uses Factory pattern for flexible provider selection.

    Args:
        provider: LLM provider name ('openai', 'gemini', 'local', 'openrouter').
                 If None, uses settings.rag.strong_llm_provider

    Returns:
        LLMBase implementation for the specified provider

    Raises:
        ValueError: If provider is not supported
    """
    provider = provider or settings.rag.strong_llm_provider
    provider = provider.lower()

    if provider == "openai":
        return OpenAILLM(model_name=settings.openai.strong_llm_model_name)
    elif provider == "gemini":
        return GeminiLLM(model_name=settings.gemini.strong_llm_model_name)
    elif provider == "local":
        return LocalLLM(model_name=settings.local_llm.strong_llm_model_name)
    elif provider == "openrouter":
        return OpenRouterLLM(model_name=settings.openrouter.strong_llm_model_name)
    else:
        raise ValueError(f"Unsupported strong LLM provider: {provider}")


__all__ = [
    "LLMBase",
    "OpenAILLM",
    "GeminiLLM",
    "LocalLLM",
    "OpenRouterLLM",
    "create_llm",
    "create_strong_llm",
]
