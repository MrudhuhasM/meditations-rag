"""
Simplified settings configuration for Meditations RAG.

This module provides organized settings classes with basic configuration
for the RAG system. Advanced features have been removed for simplicity.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    """Base configuration class with common settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    @field_validator("*", mode="before")
    @classmethod
    def parse_empty_string(cls, v: Any) -> Any:
        """Convert empty strings to None for optional fields."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


class AppSettings(BaseConfig):
    """Application-level settings."""

    # Application Identity
    app_name: str = Field(
        default="meditations-rag",
        description="Application name",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Runtime environment",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API server port",
    )
    secret_key: str = Field(
        default="change-me-in-production",
        description="Secret key for cryptographic operations",
        min_length=32,
    )

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        """Validate secret key in production."""
        environment = info.data.get("environment", "development")
        if environment == "production" and v == "change-me-in-production":
            raise ValueError("SECRET_KEY must be changed from default in production")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment.lower() == "development"

    @property
    def api_url(self) -> str:
        """Get full API URL."""
        return f"http://{self.api_host}:{self.api_port}"


class LoggingSettings(BaseConfig):
    """Logging configuration settings."""

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_console_enabled: bool = Field(
        default=True,
        description="Enable console logging",
    )
    log_file_enabled: bool = Field(
        default=True,
        description="Enable file logging",
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files",
    )

    @field_validator("log_dir")
    @classmethod
    def create_log_directory(cls, v: Path) -> Path:
        """Ensure log directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def log_file_path(self) -> Path:
        """Get full path to log file."""
        return self.log_dir / "meditations_rag.log"


class OpenAISettings(BaseConfig):
    """OpenAI-specific configuration."""

    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key",
    )
    openai_organization: str | None = Field(
        default=None,
        description="OpenAI organization ID",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Default OpenAI chat model",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model",
    )
    openai_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for generation (0-2)",
    )
    openai_max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate",
    )
    openai_timeout: int = Field(
        default=60,
        ge=1,
        description="Request timeout in seconds",
    )

    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: SecretStr, info) -> SecretStr:
        """Validate API key is set in production."""
        environment = info.data.get("environment", "development")
        if environment == "production" and not v.get_secret_value():
            raise ValueError("OPENAI_API_KEY must be set in production")
        return v

    def get_api_key(self) -> str:
        """Get the API key as a string."""
        return self.openai_api_key.get_secret_value()


class GeminiSettings(BaseConfig):
    """Google Gemini-specific configuration."""

    gemini_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Google Gemini API key",
    )
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        description="Gemini API base URL",
    )
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Default Gemini model",
    )
    gemini_embedding_model: str = Field(
        default="text-embedding-004",
        description="Gemini embedding model",
    )
    gemini_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for generation (0-2)",
    )
    gemini_max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate",
    )
    gemini_timeout: int = Field(
        default=60,
        ge=1,
        description="Request timeout in seconds",
    )

    def get_api_key(self) -> str:
        """Get the API key as a string."""
        return self.gemini_api_key.get_secret_value()


class LocalLLMSettings(BaseConfig):
    """Local LLM server configuration (OpenAI-compatible API)."""

    local_llm_base_url: str = Field(
        default="http://localhost:8080/v1",
        description="Local LLM API base URL (OpenAI-compatible)",
    )

    local_llm_embedding_url: str = Field(
        default="http://localhost:8080/v1/embeddings",
        description="Local LLM Embedding API URL (OpenAI-compatible)",
    )
    
    local_llm_api_key: str = Field(
        default="not-needed",
        description="API key for local LLM (often not required)",
    )
    local_llm_model: str = Field(
        default="local-model",
        description="Model name/identifier for local LLM",
    )
    local_llm_embedding_model: str = Field(
        default="local-embedding",
        description="Embedding model name for local LLM",
    )
    local_llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for generation (0-2)",
    )
    local_llm_max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate",
    )
    local_llm_timeout: int = Field(
        default=120,
        ge=1,
        description="Request timeout in seconds (longer for local)",
    )
    local_llm_context_length: int = Field(
        default=4096,
        ge=1,
        description="Maximum context length supported",
    )

    @field_validator("local_llm_base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Ensure base URL doesn't end with slash."""
        return v.rstrip("/")


class LLMSettings(BaseConfig):
    """Unified LLM configuration with provider selection."""

    # Provider Selection
    llm_provider: Literal["openai", "gemini", "local"] = Field(
        default="openai",
        description="Primary LLM provider to use",
    )
    embedding_provider: Literal["openai", "gemini", "local"] = Field(
        default="openai",
        description="Embedding provider to use",
    )

    # Provider-specific settings (composed)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    local: LocalLLMSettings = Field(default_factory=LocalLLMSettings)

    @field_validator("llm_provider", "embedding_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Normalize provider name."""
        return v.lower()

    def get_active_llm_config(self) -> OpenAISettings | GeminiSettings | LocalLLMSettings:
        """Get configuration for the active LLM provider."""
        if self.llm_provider == "openai":
            return self.openai
        elif self.llm_provider == "gemini":
            return self.gemini
        elif self.llm_provider == "local":
            return self.local
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

    def get_active_embedding_config(self) -> OpenAISettings | GeminiSettings | LocalLLMSettings:
        """Get configuration for the active embedding provider."""
        if self.embedding_provider == "openai":
            return self.openai
        elif self.embedding_provider == "gemini":
            return self.gemini
        elif self.embedding_provider == "local":
            return self.local
        else:
            raise ValueError(f"Unknown embedding provider: {self.embedding_provider}")


class VectorDBSettings(BaseConfig):
    """Vector database settings."""

    # Vector Database (Qdrant only)
    qdrant_mode: Literal["local", "cloud", "server"] = Field(
        default="local",
        description="Qdrant deployment mode",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    qdrant_api_key: SecretStr | None = Field(
        default=None,
        description="Qdrant API key",
    )
    qdrant_collection_name: str = Field(
        default="meditations",
        description="Qdrant collection name",
    )
    qdrant_vector_size: int = Field(
        default=1536,
        ge=1,
        description="Vector dimension size",
    )

    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, v: str) -> str:
        """Ensure URL doesn't end with slash."""
        return v.rstrip("/")

    @field_validator("qdrant_api_key")
    @classmethod
    def validate_qdrant_api_key(cls, v: SecretStr | None, info) -> SecretStr | None:
        """Validate API key is set for cloud mode in production."""
        mode = info.data.get("qdrant_mode", "local")
        environment = info.data.get("environment", "development")
        if mode == "cloud" and environment == "production" and not v:
            raise ValueError("QDRANT_API_KEY must be set for cloud mode in production")
        return v

    def get_qdrant_api_key(self) -> str | None:
        """Get the Qdrant API key as a string."""
        if self.qdrant_api_key:
            return self.qdrant_api_key.get_secret_value()
        return None


class RAGSettings(BaseConfig):
    """RAG pipeline settings."""

    # RAG Pipeline
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Target size for text chunks",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of documents to retrieve",
    )
    system_prompt: str = Field(
        default="You are a helpful assistant that answers questions based on the provided context from Marcus Aurelius's Meditations.",
        description="System prompt for the LLM",
    )
    prompt_template: str = Field(
        default="""Context:
{context}

Question: {question}

Answer based on the context above. If the answer cannot be found in the context, say so.""",
        description="Template for constructing prompts",
    )
    rag_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation",
    )
    rag_max_tokens: int = Field(
        default=500,
        ge=1,
        description="Maximum tokens in generated response",
    )


class Settings(BaseConfig):
    """Unified application settings."""

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

    def get_config_summary(self) -> dict[str, Any]:
        """
        Get a summary of current configuration.

        Returns:
            Dictionary containing non-sensitive configuration information
        """
        active_llm = self.llm.get_active_llm_config()
        model_name = getattr(active_llm, 'openai_model', getattr(active_llm, 'gemini_model', getattr(active_llm, 'local_llm_model', 'unknown')))

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
                "model": model_name,
            },
            "vectordb": {
                "provider": "qdrant",
                "mode": self.vectordb.qdrant_mode,
                "collection": self.vectordb.qdrant_collection_name,
            },
            "rag": {
                "chunk_size": self.rag.chunk_size,
                "top_k": self.rag.top_k,
            },
            "logging": {
                "level": self.logging.log_level,
                "console_enabled": self.logging.log_console_enabled,
                "file_enabled": self.logging.log_file_enabled,
            },
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance (Singleton pattern).

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
    "OpenAISettings",
    "GeminiSettings",
    "LocalLLMSettings",
]
