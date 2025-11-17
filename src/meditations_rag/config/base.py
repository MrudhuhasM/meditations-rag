"""
Base settings configuration.

This module provides the base configuration class that all other
settings modules inherit from, ensuring consistent behavior across
the application.
"""

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseAppSettings(BaseSettings):
    """Base settings class with common configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",  # Allows nested config via APP__SETTING
    )

    @field_validator("*", mode="before")
    @classmethod
    def parse_empty_string(cls, v: Any) -> Any:
        """Convert empty strings to None for optional fields."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @classmethod
    def create_directory_validator(cls, field_name: str):
        """Factory for creating directory validators."""
        @field_validator(field_name)
        @classmethod
        def validate_directory(cls, v: Path) -> Path:
            """Ensure directory exists."""
            if v:
                v.mkdir(parents=True, exist_ok=True)
            return v
        return validate_directory


class EnvironmentMixin:
    """Mixin for environment-related properties."""

    environment: str

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment.lower() == "development"

    @property
    def is_staging(self) -> bool:
        """Check if running in staging."""
        return self.environment.lower() == "staging"
