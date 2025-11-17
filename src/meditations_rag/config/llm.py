"""
LLM provider configuration settings.

This module contains settings for different LLM providers including
OpenAI, Google Gemini, and local LLM servers.
"""

from typing import Literal

from pydantic import Field, SecretStr, field_validator

from meditations_rag.config.base import BaseAppSettings


class OpenAISettings(BaseAppSettings):
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
    openai_max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries",
    )
    openai_streaming: bool = Field(
        default=False,
        description="Enable streaming responses",
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


class GeminiSettings(BaseAppSettings):
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
    gemini_max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries",
    )
    gemini_safety_settings: dict[str, str] = Field(
        default_factory=dict,
        description="Safety settings for content filtering",
    )

    def get_api_key(self) -> str:
        """Get the API key as a string."""
        return self.gemini_api_key.get_secret_value()


class LocalLLMSettings(BaseAppSettings):
    """Local LLM server configuration (OpenAI-compatible API)."""

    local_llm_enabled: bool = Field(
        default=False,
        description="Enable local LLM",
    )
    local_llm_base_url: str = Field(
        default="http://localhost:8080/v1",
        description="Local LLM API base URL (OpenAI-compatible)",
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


class LLMSettings(BaseAppSettings):
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

    # Common Settings
    llm_default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature across providers",
    )
    llm_request_timeout: int = Field(
        default=60,
        ge=1,
        description="Default request timeout in seconds",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        description="Default maximum retries",
    )

    # Rate Limiting
    llm_rate_limit_requests: int = Field(
        default=100,
        ge=1,
        description="Maximum requests per minute",
    )
    llm_rate_limit_tokens: int = Field(
        default=100000,
        ge=1,
        description="Maximum tokens per minute",
    )

    # Caching
    llm_cache_enabled: bool = Field(
        default=True,
        description="Enable response caching",
    )
    llm_cache_ttl: int = Field(
        default=3600,
        ge=0,
        description="Cache time-to-live in seconds",
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
