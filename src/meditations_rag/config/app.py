"""
Application-level settings.

This module contains core application configuration including
environment, versioning, and runtime settings.
"""

from typing import Literal

from pydantic import Field, field_validator

from meditations_rag.config.base import BaseAppSettings, EnvironmentMixin


class AppSettings(BaseAppSettings, EnvironmentMixin):
    """Core application settings."""

    # Application Identity
    app_name: str = Field(
        default="meditations-rag",
        description="Application name",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version",
    )
    app_description: str = Field(
        default="RAG system for Meditations by Marcus Aurelius",
        description="Application description",
    )

    # Runtime Environment
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
    api_prefix: str = Field(
        default="/api/v1",
        description="API route prefix",
    )
    api_title: str = Field(
        default="Meditations RAG API",
        description="API documentation title",
    )

    # Security
    secret_key: str = Field(
        default="change-me-in-production",
        description="Secret key for cryptographic operations",
        min_length=32,
    )
    allowed_hosts: list[str] = Field(
        default=["localhost", "127.0.0.1"],
        description="Allowed hosts for the application",
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins",
    )
    cors_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests",
    )
    cors_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed HTTP methods for CORS",
    )

    # Database
    database_url: str = Field(
        default="sqlite:///./data/meditations.db",
        description="Database connection URL",
    )
    database_echo: bool = Field(
        default=False,
        description="Echo SQL queries (development only)",
    )
    database_pool_size: int = Field(
        default=5,
        ge=1,
        description="Database connection pool size",
    )
    database_max_overflow: int = Field(
        default=10,
        ge=0,
        description="Maximum overflow for connection pool",
    )

    # Performance
    workers: int = Field(
        default=1,
        ge=1,
        description="Number of worker processes",
    )
    worker_timeout: int = Field(
        default=300,
        ge=30,
        description="Worker timeout in seconds",
    )
    max_requests: int = Field(
        default=1000,
        ge=0,
        description="Max requests per worker (0 = unlimited)",
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Normalize environment value."""
        return v.lower()

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        """Validate secret key in production."""
        # Access environment through info.data
        environment = info.data.get("environment", "development")
        if environment == "production" and v == "change-me-in-production":
            raise ValueError(
                "SECRET_KEY must be changed from default in production"
            )
        return v

    @property
    def api_url(self) -> str:
        """Get full API URL."""
        return f"http://{self.api_host}:{self.api_port}{self.api_prefix}"

    def model_post_init(self, __context) -> None:
        """Post-initialization validation."""
        if self.is_production:
            # Production-specific validations
            if self.debug:
                raise ValueError("DEBUG must be False in production")
            if "localhost" in self.allowed_hosts or "127.0.0.1" in self.allowed_hosts:
                raise ValueError(
                    "Remove localhost from ALLOWED_HOSTS in production"
                )
