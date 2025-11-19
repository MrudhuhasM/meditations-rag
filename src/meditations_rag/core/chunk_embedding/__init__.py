"""
Chunk Embedding Factory.

This module provides a factory function to create LlamaIndex embedding models
used specifically for the chunking process (SemanticSplitterNodeParser).
"""

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from meditations_rag.config import settings, get_logger

logger = get_logger(__name__)


def get_chunk_embedding_model() -> OpenAIEmbedding | GoogleGenAIEmbedding:
    """
    Get the chunk embedding model based on configuration.

    This model is used by LlamaIndex's SemanticSplitterNodeParser to
    determine semantic boundaries for chunking.

    Returns:
        OpenAIEmbedding | GoogleGenAIEmbedding: Configured embedding model instance

    Raises:
        ValueError: If provider is not supported or API keys are missing
    """
    provider = settings.rag.embedding_provider.lower()

    if provider == "openai":
        try:
            if settings.openai.api_key is None:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")

            logger.info(
                f"Using OpenAI embedding model for Chunking: {settings.openai.embedding_model_name}"
            )
            return OpenAIEmbedding(
                model=settings.openai.embedding_model_name,
                api_key=settings.openai.api_key.get_secret_value(),
                timeout=settings.openai.timeout,
                max_retries=settings.openai.max_retries,
                embed_batch_size=settings.rag.chunk_embed_batch_size,
            )
        except Exception as e:
            logger.error(f"Failed to create OpenAI chunk embedding model: {e}")
            raise

    elif provider == "gemini":
        try:
            if settings.gemini.api_key is None:
                raise ValueError("Gemini API key is required for Gemini embeddings")

            logger.info(
                f"Using Gemini embedding model for Chunking: {settings.gemini.embedding_model_name}"
            )
            return GoogleGenAIEmbedding(
                model_name=settings.gemini.embedding_model_name,
                api_key=settings.gemini.api_key.get_secret_value(),
            )
        except Exception as e:
            logger.error(f"Failed to create Gemini chunk embedding model: {e}")
            raise

    elif provider == "local":
        try:
            api_key_value = (
                settings.local_llm.api_key.get_secret_value()
                if settings.local_llm.api_key
                else "not-needed"
            )
            logger.info(
                f"Using Local LLM embedding model for Chunking: {settings.local_llm.embedding_model_name}"
            )
            return OpenAIEmbedding(
                model_name=settings.local_llm.embedding_model_name,
                api_key=api_key_value,
                api_base=settings.local_llm.api_embedding_base_url,
                timeout=settings.local_llm.timeout,
                max_retries=settings.local_llm.max_retries,
                embed_batch_size=settings.rag.chunk_embed_batch_size,
            )
        except Exception as e:
            logger.error(f"Failed to create Local LLM chunk embedding model: {e}")
            raise

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")