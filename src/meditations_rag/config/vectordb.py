"""
Vector database configuration settings.

This module contains settings for vector database providers,
primarily Qdrant, with support for both local and cloud deployments.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator

from meditations_rag.config.base import BaseAppSettings


class QdrantSettings(BaseAppSettings):
    """Qdrant vector database configuration."""

    # Connection Mode
    qdrant_mode: Literal["local", "cloud", "server"] = Field(
        default="local",
        description="Qdrant deployment mode",
    )

    # Local Mode Settings
    qdrant_path: Path = Field(
        default=Path("data/vector_db/qdrant"),
        description="Path for local Qdrant storage",
    )

    # Server/Cloud Mode Settings
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    qdrant_api_key: SecretStr | None = Field(
        default=None,
        description="Qdrant API key (for cloud or secured server)",
    )
    qdrant_https: bool = Field(
        default=False,
        description="Use HTTPS for Qdrant connection",
    )
    qdrant_prefix: str = Field(
        default="",
        description="URL prefix for Qdrant API",
    )

    # Connection Settings
    qdrant_timeout: int = Field(
        default=30,
        ge=1,
        description="Request timeout in seconds",
    )
    qdrant_grpc_port: int = Field(
        default=6334,
        ge=1,
        le=65535,
        description="gRPC port for Qdrant",
    )
    qdrant_prefer_grpc: bool = Field(
        default=False,
        description="Prefer gRPC over HTTP when available",
    )

    # Collection Settings
    qdrant_collection_name: str = Field(
        default="meditations",
        description="Default collection name",
    )
    qdrant_vector_size: int = Field(
        default=1536,  # OpenAI text-embedding-3-small
        ge=1,
        description="Vector dimension size",
    )
    qdrant_distance_metric: Literal["Cosine", "Euclid", "Dot"] = Field(
        default="Cosine",
        description="Distance metric for similarity search",
    )

    # Performance Settings
    qdrant_batch_size: int = Field(
        default=100,
        ge=1,
        description="Batch size for bulk operations",
    )
    qdrant_parallel_requests: int = Field(
        default=4,
        ge=1,
        description="Number of parallel requests",
    )
    qdrant_max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries",
    )

    # Indexing Settings
    qdrant_on_disk: bool = Field(
        default=False,
        description="Store vectors on disk instead of RAM",
    )
    qdrant_hnsw_m: int = Field(
        default=16,
        ge=4,
        le=64,
        description="HNSW M parameter (graph connectivity)",
    )
    qdrant_hnsw_ef_construct: int = Field(
        default=100,
        ge=4,
        description="HNSW ef_construct parameter",
    )
    qdrant_hnsw_full_scan_threshold: int = Field(
        default=10000,
        ge=1,
        description="Threshold for full scan vs HNSW",
    )

    # Optimization Settings
    qdrant_optimize_interval: int = Field(
        default=3600,
        ge=0,
        description="Optimization interval in seconds (0=disabled)",
    )
    qdrant_optimize_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Optimization threshold (fraction of deleted vectors)",
    )

    @field_validator("qdrant_path")
    @classmethod
    def create_qdrant_directory(cls, v: Path) -> Path:
        """Ensure Qdrant directory exists for local mode."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("qdrant_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL doesn't end with slash."""
        return v.rstrip("/")

    @field_validator("qdrant_api_key")
    @classmethod
    def validate_api_key(cls, v: SecretStr | None, info) -> SecretStr | None:
        """Validate API key is set for cloud mode in production."""
        mode = info.data.get("qdrant_mode", "local")
        environment = info.data.get("environment", "development")

        if mode == "cloud" and environment == "production" and not v:
            raise ValueError("QDRANT_API_KEY must be set for cloud mode in production")
        return v

    def get_api_key(self) -> str | None:
        """Get the API key as a string."""
        if self.qdrant_api_key:
            return self.qdrant_api_key.get_secret_value()
        return None

    @property
    def is_local(self) -> bool:
        """Check if using local Qdrant."""
        return self.qdrant_mode == "local"

    @property
    def is_cloud(self) -> bool:
        """Check if using Qdrant Cloud."""
        return self.qdrant_mode == "cloud"

    @property
    def is_server(self) -> bool:
        """Check if using self-hosted Qdrant server."""
        return self.qdrant_mode == "server"


class VectorDBSettings(BaseAppSettings):
    """Unified vector database configuration."""

    # Provider Selection
    vectordb_provider: Literal["qdrant", "chroma", "pinecone", "weaviate"] = Field(
        default="qdrant",
        description="Vector database provider",
    )

    # Common Settings
    vectordb_batch_size: int = Field(
        default=100,
        ge=1,
        description="Default batch size for operations",
    )
    vectordb_timeout: int = Field(
        default=30,
        ge=1,
        description="Default timeout in seconds",
    )

    # Search Settings
    search_top_k: int = Field(
        default=5,
        ge=1,
        description="Default number of results to return",
    )
    search_score_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold",
    )
    search_filter_enabled: bool = Field(
        default=True,
        description="Enable metadata filtering in search",
    )

    # Provider-specific settings
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)

    @field_validator("vectordb_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Normalize provider name."""
        return v.lower()

    def get_active_config(self) -> QdrantSettings:
        """Get configuration for the active vector DB provider."""
        if self.vectordb_provider == "qdrant":
            return self.qdrant
        else:
            raise ValueError(
                f"Unsupported vector DB provider: {self.vectordb_provider}"
            )
