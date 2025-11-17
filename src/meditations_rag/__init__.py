from meditations_rag.config import settings
from meditations_rag.config.logger import get_logger, setup_logging

# Initialize logging on module import
setup_logging()


def main() -> None:
    """Main entry point for the application."""
    log = get_logger(__name__)

    log.info("Starting Meditations RAG application")
    log.info(f"Version: {settings.app.app_version}")
    log.info(f"Environment: {settings.app.environment}")

    # Print configuration summary
    config_summary = settings.get_config_summary()
    log.debug(f"Configuration: {config_summary}")

    print("Hello from meditations-rag!")

    log.info("Application startup complete")
