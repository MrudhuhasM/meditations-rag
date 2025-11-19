from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr, field_validator
from functools import lru_cache
from urllib.parse import urlparse


class AppSettings(BaseSettings):
    """Application settings for Meditations RAG."""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore", env_prefix="APP_")

    app_name: str = Field(default="Meditations RAG", description="Name of the application")
    app_version: str = Field(default="0.1.0", description="Version of the application")
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    debug: bool = Field(default=False, description="Enable debug mode")
    host: str = Field(default="0.0.0.0", description="Host to bind the server to")
    port: int = Field(default=8000, description="Port to bind the server to")
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://localhost:8080"], description="Allowed CORS origins")

    @field_validator('port')
    @classmethod
    def validate_port(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError('Port must be between 1 and 65535')
        return v

    @field_validator('cors_origins')
    @classmethod
    def validate_cors_origins(cls, v):
        for origin in v:
            parsed = urlparse(origin)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f'Invalid CORS origin URL: {origin}')
        return v


class RagSettings(BaseSettings):
    """RAG settings for Meditations RAG."""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore", env_prefix="RAG_")

    embedding_provider: str = Field(default="openai", description="Embedding service provider (e.g., openai, gemini, local)")
    llm_provider: str = Field(default="openai", description="LLM service provider (e.g., openai, gemini, local)")
    strong_llm_provider: str = Field(default="openai", description="Strong LLM service provider for complex tasks")
    buffer_size: int = Field(default=5, description="Size of the context buffer for RAG")
    break_point_threshold: int = Field(default=95 , description="Threshold for breaking points in document retrieval")
    batch_size: int = Field(default=32, description="Batch size for processing documents")
    chunk_embed_batch_size: int = Field(default=32, description="Batch size for embedding chunks")
    embedding_api_batch_size: int = Field(default=32, description="Maximum batch size for raw embedding API calls (API limit)")
    max_concurrent_requests: int = Field(default=5, description="Maximum concurrent requests to LLM/embedding services")
    failure_threshold: float = Field(default=0.2, description="percentage of allowed failures before aborting operations")
    embedding_dimension: int  = Field(default=1024, description="Dimension of the embedding vectors (if known)")
    
    # Metadata extraction settings
    metadata_extraction_enabled: bool = Field(default=True, description="Enable metadata extraction from chunks")
    metadata_batch_size: int = Field(default=10, description="Batch size for metadata extraction")
    metadata_max_concurrent: int = Field(default=5, description="Max concurrent metadata extraction requests")
    
    # Retrieval settings
    retrieval_top_k: int = Field(default=6, description="Number of results to return from retrieval (increased for metadata-aware)")
    retrieval_question_top_k: int = Field(default=10, description="Number of results to fetch from question collection")
    retrieval_alpha: float = Field(default=0.3, description="Weight for question scores in fusion (0.0-1.0)")
    retrieval_metadata_top_k: int = Field(default=5, description="Number of results per metadata filter")
    retrieval_metadata_enabled: bool = Field(default=True, description="Enable metadata-aware retrieval")
    retrieval_keyword_boost: float = Field(default=0.05, description="Score boost per keyword match")
    retrieval_topic_boost: float = Field(default=0.15, description="Score boost for topic match")
    retrieval_concept_boost: float = Field(default=0.10, description="Score boost for concept match")
    retrieval_practice_boost: float = Field(default=0.10, description="Score boost for practice match")
    
    @field_validator('retrieval_alpha')
    @classmethod
    def validate_retrieval_alpha(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('Retrieval alpha must be between 0.0 and 1.0')
        return v


class LoggingSettings(BaseSettings):
    """Logging settings."""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore", env_prefix="LOGGING_")

    log_console_enabled: bool = Field(default=True, description="Enable console logging")
    log_file_enabled: bool = Field(default=False, description="Enable file logging")
    log_level: str = Field(default="INFO", description="Logging level")
    log_file_path: str = Field(default="logs/app.log", description="Path to log file")


class OpenAISettings(BaseSettings):
    """OpenAI API settings."""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore", env_prefix="OPENAI_")

    api_key: SecretStr | None = Field(default=None, description="OpenAI API key")
    api_base: str = Field(default="https://api.openai.com/v1", description="Base URL for OpenAI API")
    llm_model_name: str = Field(default="gpt-4o-mini", description="Default OpenAI model name")
    embedding_model_name: str = Field(default="text-embedding-ada-002", description="Default OpenAI embedding model name")
    timeout: int = Field(default=30, description="Timeout for OpenAI API requests in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries for OpenAI API requests")
    max_tokens: int = Field(default=2048, description="Maximum tokens for LLM responses")
    temperature: float = Field(default=0.7, description="Temperature setting for LLM responses")
    reasoning_enabled: bool = Field(default=True, description="Enable reasoning capabilities for LLM")
    
    # Rate limiting settings
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting for OpenAI API")
    rate_limit_requests_per_minute: int = Field(default=500, description="Max requests per minute (tier-based, adjust per your tier)")
    rate_limit_tokens_per_minute: int = Field(default=150000, description="Max tokens per minute (tier-based, adjust per your tier)")

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        if v is not None and not v.get_secret_value().strip():
            raise ValueError('OpenAI API key cannot be empty')
        return v

    @field_validator('api_base')
    @classmethod
    def validate_api_base(cls, v):
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError('Invalid OpenAI API base URL')
        return v

    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError('Timeout must be positive')
        return v

    @field_validator('max_retries')
    @classmethod
    def validate_max_retries(cls, v):
        if v < 0:
            raise ValueError('Max retries must be non-negative')
        return v

class GeminiSettings(BaseSettings):
    """Gemini API settings."""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore", env_prefix="GEMINI_")

    api_key: SecretStr | None = Field(default=None, description="Gemini API key")
    api_base: str = Field(default="https://gemini.googleapis.com/v1", description="Base URL for Gemini API")
    llm_model_name: str = Field(default="gemini-1.5", description="Default Gemini model name")
    embedding_model_name: str = Field(default="models/text-embedding-004", description="Default Gemini embedding model name (newer model)")
    timeout: int = Field(default=30, description="Timeout for Gemini API requests in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries for Gemini API requests")
    max_tokens: int = Field(default=2048, description="Maximum tokens for LLM responses")
    temperature: float = Field(default=0.7, description="Temperature setting for LLM responses")
    reasoning_enabled: bool = Field(default=True, description="Enable reasoning capabilities for LLM")
    
    # Rate limiting settings
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting for Gemini API")
    rate_limit_requests_per_minute: int = Field(default=15, description="Max requests per minute (free tier default)")
    rate_limit_requests_per_day: int = Field(default=1500, description="Max requests per day (free tier default)")

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        if v is not None and not v.get_secret_value().strip():
            raise ValueError('Gemini API key cannot be empty')
        return v

    @field_validator('api_base')
    @classmethod
    def validate_api_base(cls, v):
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError('Invalid Gemini API base URL')
        return v

    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError('Timeout must be positive')
        return v

    @field_validator('max_retries')
    @classmethod
    def validate_max_retries(cls, v):
        if v < 0:
            raise ValueError('Max retries must be non-negative')
        return v

class LocalLLMSettings(BaseSettings):
    """Local LLM settings."""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore", env_prefix="LOCAL_LLM_")

    api_base_url: str | None = Field(default=None, description="Base URL for local LLM service")
    api_embedding_base_url: str | None = Field(default=None, description="Base URL for local embedding service")
    api_key : SecretStr | None = Field(default=None, description="API key for local LLM service if required")
    llm_model_name: str | None = Field(default=None, description="Name of the local LLM model")
    embedding_model_name: str | None = Field(default=None, description="Name of the local embedding model")
    timeout: int = Field(default=60, description="Timeout for local LLM requests in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries for local LLM requests")
    max_tokens: int = Field(default=2048, description="Maximum tokens for LLM responses")
    temperature: float = Field(default=0.7, description="Temperature setting for LLM responses")
    reasoning_enabled: bool = Field(default=True, description="Enable reasoning capabilities for LLM")

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        if v is not None and not v.get_secret_value().strip():
            raise ValueError('Local LLM API key cannot be empty')
        return v

    @field_validator('api_base_url')
    @classmethod
    def validate_api_base_url(cls, v):
        if v is not None:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError('Invalid local LLM API base URL')
        return v

    @field_validator('api_embedding_base_url')
    @classmethod
    def validate_api_embedding_base_url(cls, v):
        if v is not None:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError('Invalid local embedding API base URL')
        return v

    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError('Timeout must be positive')
        return v

    @field_validator('max_retries')
    @classmethod
    def validate_max_retries(cls, v):
        if v < 0:
            raise ValueError('Max retries must be non-negative')
        return v


class OpenRouterSettings(BaseSettings):
    """OpenRouter API settings."""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore", env_prefix="OPENROUTER_")

    api_key: SecretStr | None = Field(default=None, description="OpenRouter API key")
    api_base: str = Field(default="https://openrouter.ai/api/v1", description="Base URL for OpenRouter API")
    llm_model_name: str = Field(default="openai/gpt-4o-mini", description="Default OpenRouter model name (format: provider/model)")
    timeout: int = Field(default=30, description="Timeout for OpenRouter API requests in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries for OpenRouter API requests")
    max_tokens: int = Field(default=2048, description="Maximum tokens for LLM responses")
    temperature: float = Field(default=0.7, description="Temperature setting for LLM responses")
    
    # OpenRouter-specific headers for app attribution (optional)
    http_referer: str | None = Field(default=None, description="HTTP Referer for OpenRouter rankings (optional)")
    x_title: str | None = Field(default=None, description="App title for OpenRouter rankings (optional)")
    
    # Rate limiting settings
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting for OpenRouter API")
    rate_limit_requests_per_minute: int = Field(default=200, description="Max requests per minute (adjust based on your usage)")
    rate_limit_tokens_per_minute: int = Field(default=100000, description="Max tokens per minute (adjust based on your usage)")

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        if v is not None and not v.get_secret_value().strip():
            raise ValueError('OpenRouter API key cannot be empty')
        return v

    @field_validator('api_base')
    @classmethod
    def validate_api_base(cls, v):
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError('Invalid OpenRouter API base URL')
        return v

    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError('Timeout must be positive')
        return v

    @field_validator('max_retries')
    @classmethod
    def validate_max_retries(cls, v):
        if v < 0:
            raise ValueError('Max retries must be non-negative')
        return v


class QdrantSettings(BaseSettings):
    """Qdrant vector database settings."""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore", env_prefix="QDRANT_")

    host: str = Field(default="localhost", description="Qdrant host")
    port: int = Field(default=6333, description="Qdrant port")
    api_key: SecretStr | None = Field(default=None, description="Qdrant API key if required")
    main_collection_name: str = Field(default="meditations_collection", description="Name of the Qdrant collection to use")
    question_collection_name: str = Field(default="meditations_questions", description="Name of the Qdrant collection for questions")

    @field_validator('port')
    @classmethod
    def validate_port(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError('Qdrant port must be between 1 and 65535')
        return v

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        if v is not None and not v.get_secret_value().strip():
            raise ValueError('Qdrant API key cannot be empty')
        return v

    @field_validator('host')
    @classmethod
    def validate_host(cls, v):
        import ipaddress
        import socket
        try:
            ipaddress.ip_address(v)
        except ValueError:
            try:
                socket.gethostbyname(v)
            except socket.gaierror:
                raise ValueError('Invalid Qdrant host: must be a valid IP address or hostname')
        return v


class Settings:
    """Aggregate application settings."""

    app: AppSettings = AppSettings()
    rag: RagSettings = RagSettings()
    logging: LoggingSettings = LoggingSettings()
    openai: OpenAISettings = OpenAISettings()
    gemini: GeminiSettings = GeminiSettings()
    local_llm: LocalLLMSettings = LocalLLMSettings()
    openrouter: OpenRouterSettings = OpenRouterSettings()
    qdrant: QdrantSettings = QdrantSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the application settings."""
    return Settings()

settings = get_settings()


    