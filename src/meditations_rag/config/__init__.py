"""
Configuration package for Meditations RAG.

This package provides simplified configuration for the RAG system.

Usage:
    >>> from meditations_rag.config import settings
    >>> print(settings.app.app_name)
    >>> print(settings.llm.openai.openai_model)
"""

from meditations_rag.config.logger import logger, get_logger, setup_logging
from meditations_rag.config.settings import Settings, get_settings, settings

__all__ = [
    # Main settings instance
    "settings",
    "get_settings",
    "Settings",
    # Logger
    "setup_logging",
    "get_logger",
    "logger",
]
