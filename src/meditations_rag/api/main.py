from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from meditations_rag.config import settings, get_logger
from meditations_rag.api.routers import rag, agentic, health

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for the FastAPI application.
    """
    logger.info("Starting Meditations RAG API")
    # Perform startup tasks here (e.g., warm up caches, check DB connection)
    yield
    logger.info("Shutting down Meditations RAG API")
    # Perform shutdown tasks here

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title=settings.app.app_name,
        version=settings.app.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router)
    app.include_router(rag.router)
    app.include_router(agentic.router)

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "meditations_rag.api.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug
    )
