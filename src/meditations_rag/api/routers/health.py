from fastapi import APIRouter

from meditations_rag.api.schemas.rag import HealthResponse
from meditations_rag.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    """
    return HealthResponse(
        status="ok",
        version=settings.app.app_version,
        environment=settings.app.environment,
    )
