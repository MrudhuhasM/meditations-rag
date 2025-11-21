import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from meditations_rag.api.limiter import limiter
from meditations_rag.api.routers import agentic, health, rag
from meditations_rag.api.security import get_api_key
from meditations_rag.config import get_logger, settings

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

    # Set up Rate Limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

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
    # Protect RAG and Agentic endpoints with API Key
    app.include_router(rag.router, dependencies=[Depends(get_api_key)])
    app.include_router(agentic.router, dependencies=[Depends(get_api_key)])

    # Mount static files
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/")
        async def read_root():
            return FileResponse(os.path.join(static_dir, "index.html"))

    else:
        logger.warning(f"Static directory not found at {static_dir}")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "meditations_rag.api.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )
