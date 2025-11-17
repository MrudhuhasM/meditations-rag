"""
Logging configuration using Loguru.

This module sets up production-ready logging with rotation, retention,
and environment-specific formatting. It integrates with the settings
module to provide a fully configurable logging system.
"""

import sys
from typing import Any

from loguru import logger

from meditations_rag.config.settings import settings


def setup_logging() -> None:
    """
    Configure loguru logger with production-ready settings.

    Features:
    - Console output with colored formatting
    - File output with rotation and retention
    - Environment-specific log levels
    - Optional JSON serialization for log aggregation
    - Automatic compression of rotated logs
    - Thread-safe logging
    - Separate error log file
    """
    # Remove default handler
    logger.remove()

    # Console handler (if enabled)
    if settings.logging.log_console_enabled:
        logger.add(
            sys.stderr,
            format=settings.logging.get_format(),
            level=settings.logging.console_level,
            colorize=settings.logging.log_console_colorize,
            backtrace=settings.logging.log_backtrace,
            diagnose=settings.logging.log_diagnose,
            enqueue=settings.logging.log_enqueue,
        )

    # File handler for all logs (if enabled)
    if settings.logging.log_file_enabled:
        logger.add(
            settings.logging.log_file_path,
            format=settings.logging.get_format(),
            level=settings.logging.file_level,
            rotation=settings.logging.log_rotation,
            retention=settings.logging.log_retention,
            compression=settings.logging.log_compression,
            serialize=settings.logging.log_format_json,
            backtrace=settings.logging.log_backtrace,
            diagnose=settings.logging.log_diagnose,
            enqueue=settings.logging.log_enqueue,
        )

        # Separate error log file
        logger.add(
            settings.logging.error_log_path,
            format=settings.logging.get_format(),
            level="ERROR",
            rotation=settings.logging.log_rotation,
            retention=settings.logging.log_retention,
            compression=settings.logging.log_compression,
            serialize=settings.logging.log_format_json,
            backtrace=True,
            diagnose=True,
            enqueue=settings.logging.log_enqueue,
        )

    # Suppress verbose logging from third-party libraries
    for library in settings.logging.log_suppress_libraries:
        logger.disable(library)

    # Log initialization
    logger.info(
        f"Logging initialized for {settings.app.app_name} v{settings.app.app_version}"
    )
    logger.info(f"Environment: {settings.app.environment}")
    logger.info(f"Log level: {settings.logging.log_level}")
    logger.info(f"Log directory: {settings.logging.log_dir}")

    if settings.app.is_production:
        logger.warning("Running in PRODUCTION mode")
    elif settings.app.is_staging:
        logger.info("Running in STAGING mode")
    else:
        logger.info("Running in DEVELOPMENT mode")


def get_logger(name: str | None = None):
    """
    Get a logger instance with optional name binding.

    Args:
        name: Optional name to bind to the logger (e.g., module name)

    Returns:
        Configured logger instance

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("Processing started")
    """
    if name:
        return logger.bind(name=name)
    return logger


class LogLevel:
    """Context manager for temporary log level changes."""

    def __init__(self, level: str):
        """
        Initialize with target log level.

        Args:
            level: Target log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.level = level
        self.handler_id = None
        self.original_handlers = []

    def __enter__(self):
        """Enter context and change log level."""
        # Store current handlers
        self.original_handlers = logger._core.handlers.copy()

        # Remove all handlers
        logger.remove()

        # Add temporary handler with new level
        self.handler_id = logger.add(
            sys.stderr,
            format=settings.logging.get_format(),
            level=self.level,
            colorize=settings.logging.log_console_colorize,
        )
        return logger

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and restore original log level."""
        logger.remove(self.handler_id)
        setup_logging()


def log_execution(func):
    """
    Decorator to log function execution with timing.

    Example:
        >>> @log_execution
        ... def process_data(data):
        ...     return transformed_data
    """
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_logger = get_logger(func.__module__)
        func_logger.info(f"Starting {func.__name__}")
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            func_logger.info(f"Completed {func.__name__} in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            func_logger.error(f"Failed {func.__name__} after {elapsed:.2f}s: {e}")
            raise

    return wrapper


def log_async_execution(func):
    """
    Decorator to log async function execution with timing.

    Example:
        >>> @log_async_execution
        ... async def fetch_data(url):
        ...     return await response.json()
    """
    import functools
    import time

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        func_logger = get_logger(func.__module__)
        func_logger.info(f"Starting {func.__name__}")
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            func_logger.info(f"Completed {func.__name__} in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            func_logger.error(f"Failed {func.__name__} after {elapsed:.2f}s: {e}")
            raise

    return wrapper


def log_context(**context_vars):
    """
    Decorator to add context variables to log messages.

    Example:
        >>> @log_context(service="api", version="v1")
        ... def handle_request():
        ...     log = get_logger(__name__)
        ...     log.info("Processing request")
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with logger.contextualize(**context_vars):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def catch_exceptions(reraise: bool = True, default: Any = None):
    """
    Decorator to catch and log exceptions.

    Args:
        reraise: Whether to reraise the exception after logging
        default: Default value to return if exception occurs and reraise=False

    Example:
        >>> @catch_exceptions(reraise=False, default=None)
        ... def risky_operation():
        ...     return 1 / 0
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                func_logger = get_logger(func.__module__)
                func_logger.exception(f"Exception in {func.__name__}: {e}")
                if reraise:
                    raise
                return default

        return wrapper

    return decorator


# Export commonly used items
__all__ = [
    "setup_logging",
    "get_logger",
    "logger",
    "LogLevel",
    "log_execution",
    "log_async_execution",
    "log_context",
    "catch_exceptions",
]
