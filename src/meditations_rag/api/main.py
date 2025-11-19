from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

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
        reload=settings.app.debug
    )
