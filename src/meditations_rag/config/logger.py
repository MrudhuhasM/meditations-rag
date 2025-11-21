"""
Basic logging configuration using Loguru.

This module sets up simple logging with console and file output.
"""

import sys

from loguru import logger

from meditations_rag.config.settings import settings


def setup_logging() -> None:
    """
    Configure loguru logger with basic settings.
    """
    # Remove default handler
    logger.remove()

    # Console handler
    if settings.logging.log_console_enabled:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level=settings.logging.log_level,
            colorize=True,
        )

    # File handler
    if settings.logging.log_file_enabled:
        logger.add(
            settings.logging.log_file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level=settings.logging.log_level,
            rotation="10 MB",
            retention="30 days",
        )

    # Log initialization
    logger.info(
        f"Logging initialized for {settings.app.app_name} v{settings.app.app_version}"
    )
    logger.info(f"Environment: {settings.app.environment}")


def get_logger(name: str | None = None):
    """
    Get a logger instance with optional name binding.

    Args:
        name: Optional name to bind to the logger

    Returns:
        Configured logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Export public interface
__all__ = [
    "setup_logging",
    "get_logger",
    "logger",
]
