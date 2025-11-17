"""
Configuration package for Meditations RAG.

This package provides a modular, production-ready configuration system
following best practices and design patterns.

Usage:
    >>> from meditations_rag.config import settings
    >>> print(settings.app.app_name)
    >>> print(settings.llm.openai.openai_model)
    >>> print(settings.rag.retrieval.top_k)

Architecture:
    - base.py: Base classes and mixins for all settings
    - app.py: Application-level settings
    - logging.py: Logging configuration
    - llm.py: LLM provider settings (OpenAI, Gemini, Local)
    - vectordb.py: Vector database settings (Qdrant)
    - rag.py: RAG pipeline settings
    - settings.py: Main settings factory (composes all modules)
"""

from meditations_rag.config.settings import (
    AppSettings,
    LLMSettings,
    LoggingSettings,
    RAGSettings,
    Settings,
    VectorDBSettings,
    get_settings,
    settings,
)

__all__ = [
    # Main settings instance
    "settings",
    "get_settings",
    # Settings classes
    "Settings",
    "AppSettings",
    "LoggingSettings",
    "LLMSettings",
    "VectorDBSettings",
    "RAGSettings",
]
