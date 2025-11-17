"""
Logging configuration settings.

This module contains all logging-related configuration for the application,
including log levels, formats, rotation, and output destinations.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from meditations_rag.config.base import BaseAppSettings


class LoggingSettings(BaseAppSettings):
    """Logging configuration settings."""

    # Log Levels
    log_level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Global logging level",
    )
    log_level_console: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = Field(
        default=None,
        description="Console-specific log level (defaults to log_level)",
    )
    log_level_file: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = Field(
        default=None,
        description="File-specific log level (defaults to log_level)",
    )

    # Log Format
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Log message format (supports loguru formatting)",
    )
    log_format_json: bool = Field(
        default=False,
        description="Use JSON format for logs (recommended for production)",
    )

    # Console Output
    log_console_enabled: bool = Field(
        default=True,
        description="Enable console logging",
    )
    log_console_colorize: bool = Field(
        default=True,
        description="Colorize console output",
    )

    # File Output
    log_file_enabled: bool = Field(
        default=True,
        description="Enable file logging",
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files",
    )
    log_filename: str = Field(
        default="meditations_rag.log",
        description="Main log filename",
    )
    log_error_filename: str = Field(
        default="error.log",
        description="Error-only log filename",
    )

    # Rotation and Retention
    log_rotation: str = Field(
        default="100 MB",
        description="Log rotation threshold (size or time-based)",
    )
    log_retention: str = Field(
        default="30 days",
        description="Log file retention period",
    )
    log_compression: str = Field(
        default="zip",
        description="Compression format for rotated logs",
    )

    # Advanced Options
    log_backtrace: bool = Field(
        default=True,
        description="Include backtrace in error logs",
    )
    log_diagnose: bool = Field(
        default=False,
        description="Include variable values in error logs (disable in production)",
    )
    log_enqueue: bool = Field(
        default=True,
        description="Use thread-safe logging queue",
    )
    log_catch: bool = Field(
        default=True,
        description="Automatically catch exceptions in decorated functions",
    )

    # Contextual Logging
    log_include_request_id: bool = Field(
        default=True,
        description="Include request ID in logs",
    )
    log_include_user_id: bool = Field(
        default=True,
        description="Include user ID in logs when available",
    )

    # Third-party Library Logging
    log_library_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="WARNING",
        description="Log level for third-party libraries",
    )
    log_suppress_libraries: list[str] = Field(
        default=["urllib3", "httpx", "httpcore"],
        description="Libraries to suppress verbose logging",
    )

    @field_validator("log_dir")
    @classmethod
    def create_log_directory(cls, v: Path) -> Path:
        """Ensure log directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("log_diagnose")
    @classmethod
    def validate_diagnose(cls, v: bool, info) -> bool:
        """Disable diagnose in production for security."""
        environment = info.data.get("environment", "development")
        if environment == "production" and v:
            raise ValueError("log_diagnose must be False in production")
        return v

    @property
    def console_level(self) -> str:
        """Get effective console log level."""
        return self.log_level_console or self.log_level

    @property
    def file_level(self) -> str:
        """Get effective file log level."""
        return self.log_level_file or self.log_level

    @property
    def log_file_path(self) -> Path:
        """Get full path to main log file."""
        return self.log_dir / self.log_filename

    @property
    def error_log_path(self) -> Path:
        """Get full path to error log file."""
        return self.log_dir / self.log_error_filename

    def get_format(self) -> str:
        """Get appropriate log format based on settings."""
        if self.log_format_json:
            return "{message}"  # Structured JSON format
        return self.log_format
