from .base import LLMBase
from .openai import OpenAILLM
from .gemini import GeminiLLM
from .local_llm import LocalLLM
from meditations_rag.config import settings, get_logger


def create_llm(provider: str | None = None) -> LLMBase:
    """
    Factory function to create LLM instances based on provider.
    
    Uses Factory pattern for flexible provider selection.
    
    Args:
        provider: LLM provider name ('openai', 'gemini', 'local').
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
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


__all__ = ["LLMBase", "OpenAILLM", "GeminiLLM", "LocalLLM", "create_llm"]