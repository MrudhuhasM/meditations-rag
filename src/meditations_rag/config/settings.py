"""
Main settings factory and configuration management.

This module provides the unified Settings class that composes all
configuration modules and serves as the single source of truth for
application configuration.
"""

from functools import lru_cache
from typing import Any

from pydantic import Field

from meditations_rag.config.app import AppSettings
from meditations_rag.config.base import BaseAppSettings
from meditations_rag.config.llm import LLMSettings
from meditations_rag.config.logging import LoggingSettings
from meditations_rag.config.rag import RAGSettings
from meditations_rag.config.vectordb import VectorDBSettings


class Settings(BaseAppSettings):
    """
    Unified application settings.

    This class composes all configuration modules into a single
    settings object following the Composition design pattern.

    Usage:
        >>> from meditations_rag.config import settings
        >>> print(settings.app.app_name)
        >>> print(settings.llm.openai.openai_model)
        >>> print(settings.rag.retrieval.top_k)
    """

    # Composed settings modules
    app: AppSettings = Field(
        default_factory=AppSettings,
        description="Application-level settings",
    )
    logging: LoggingSettings = Field(
        default_factory=LoggingSettings,
        description="Logging configuration",
    )
    llm: LLMSettings = Field(
        default_factory=LLMSettings,
        description="LLM provider settings",
    )
    vectordb: VectorDBSettings = Field(
        default_factory=VectorDBSettings,
        description="Vector database settings",
    )
    rag: RAGSettings = Field(
        default_factory=RAGSettings,
        description="RAG pipeline settings",
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization validation and cross-module checks.

        This method ensures consistency across different configuration modules
        and performs environment-specific validation.
        """
        # Propagate environment to all sub-modules
        environment = self.app.environment
        self.logging.environment = environment
        self.llm.environment = environment
        self.vectordb.environment = environment
        self.rag.environment = environment

        # Production-specific validations
        if self.app.is_production:
            self._validate_production_config()

        # Cross-module consistency checks
        self._validate_cross_module_consistency()

    def _validate_production_config(self) -> None:
        """Validate configuration for production environment."""
        errors = []

        # Check critical API keys
        if not self.llm.openai.get_api_key() and self.llm.llm_provider == "openai":
            errors.append("OpenAI API key must be set in production")

        if not self.llm.gemini.get_api_key() and self.llm.llm_provider == "gemini":
            errors.append("Gemini API key must be set in production")

        # Check security settings
        if self.app.secret_key == "change-me-in-production":
            errors.append("SECRET_KEY must be changed from default")

        # Check logging settings
        if self.logging.log_diagnose:
            errors.append("log_diagnose must be disabled in production")

        if errors:
            raise ValueError(
                f"Production validation failed:\n" + "\n".join(f"- {e}" for e in errors)
            )

    def _validate_cross_module_consistency(self) -> None:
        """Validate consistency across configuration modules."""
        # Ensure vector dimensions match embedding model
        embedding_model = None
        if self.llm.embedding_provider == "openai":
            embedding_model = self.llm.openai.openai_embedding_model
            if "text-embedding-3-small" in embedding_model:
                expected_dim = 1536
            elif "text-embedding-3-large" in embedding_model:
                expected_dim = 3072
            elif "text-embedding-ada-002" in embedding_model:
                expected_dim = 1536
            else:
                return  # Unknown model, skip validation

            if self.vectordb.qdrant.qdrant_vector_size != expected_dim:
                raise ValueError(
                    f"Vector dimension mismatch: {embedding_model} produces "
                    f"{expected_dim}-dimensional vectors, but Qdrant is configured "
                    f"for {self.vectordb.qdrant.qdrant_vector_size} dimensions"
                )

    def get_config_summary(self) -> dict[str, Any]:
        """
        Get a summary of current configuration.

        Returns:
            Dictionary containing non-sensitive configuration information
        """
        return {
            "app": {
                "name": self.app.app_name,
                "version": self.app.app_version,
                "environment": self.app.environment,
                "debug": self.app.debug,
            },
            "llm": {
                "provider": self.llm.llm_provider,
                "embedding_provider": self.llm.embedding_provider,
                "model": self.llm.get_active_llm_config().openai_model
                if self.llm.llm_provider == "openai"
                else "N/A",
            },
            "vectordb": {
                "provider": self.vectordb.vectordb_provider,
                "mode": self.vectordb.qdrant.qdrant_mode
                if self.vectordb.vectordb_provider == "qdrant"
                else "N/A",
            },
            "rag": {
                "enabled": self.rag.rag_enabled,
                "mode": self.rag.rag_mode,
                "chunk_size": self.rag.document_processing.chunk_size,
                "top_k": self.rag.retrieval.top_k,
            },
            "logging": {
                "level": self.logging.log_level,
                "console_enabled": self.logging.log_console_enabled,
                "file_enabled": self.logging.log_file_enabled,
            },
        }

    def reload(self) -> "Settings":
        """
        Reload settings from environment variables.

        Returns:
            New Settings instance with refreshed values
        """
        get_settings.cache_clear()
        return get_settings()


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance (Singleton pattern).

    This function uses LRU cache to ensure only one Settings instance
    exists throughout the application lifecycle.

    Returns:
        Cached Settings instance
    """
    return Settings()


# Global settings instance
settings = get_settings()


# Export public interface
__all__ = [
    "Settings",
    "settings",
    "get_settings",
    # Sub-modules for direct import if needed
    "AppSettings",
    "LoggingSettings",
    "LLMSettings",
    "VectorDBSettings",
    "RAGSettings",
]
