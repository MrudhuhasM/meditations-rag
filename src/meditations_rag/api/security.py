from typing import Dict, Optional

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from meditations_rag.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

# Simple in-memory usage tracker for demo purposes
# Structure: { "ip_address": { "standard": 0, "agentic": 0 } }
USAGE_TRACKER: Dict[str, Dict[str, int]] = {}

FREE_TIER_LIMITS = {"standard": 4, "agentic": 2}


async def get_api_key(
    header_key: str | None = Security(api_key_header),
    query_key: str | None = Security(api_key_query),
) -> Optional[str]:
    """
    Validate API key from header or query parameter.
    Returns the key if valid, None if not provided (and not strictly required by this function alone).
    """
    api_key = header_key or query_key

    if api_key:
        if settings.app.api_key and api_key != settings.app.api_key.get_secret_value():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API Key",
            )
        return api_key

    return None


async def verify_standard_rag_access(
    request: Request, api_key: str | None = Security(get_api_key)
):
    """
    Verifies access for Standard RAG endpoints.
    Allows limited free requests per IP, then requires API key.
    """
    if api_key:
        return  # Valid API key provided, allow access

    # No API key, check free tier limits
    client_ip = request.client.host if request.client else "unknown"

    if client_ip not in USAGE_TRACKER:
        USAGE_TRACKER[client_ip] = {"standard": 0, "agentic": 0}

    usage = USAGE_TRACKER[client_ip]["standard"]

    if usage >= FREE_TIER_LIMITS["standard"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Since this is a demo i built for my portfolio, i have set limits ({FREE_TIER_LIMITS['standard']} requests) to prevent abuse. If i have proided you an API Key, please use it to continue accessing the Standard RAG endpoints.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Increment usage
    USAGE_TRACKER[client_ip]["standard"] += 1
    return


async def verify_agentic_rag_access(
    request: Request, api_key: str | None = Security(get_api_key)
):
    """
    Verifies access for Agentic RAG endpoints.
    Allows limited free requests per IP, then requires API key.
    """
    if api_key:
        return  # Valid API key provided, allow access

    # No API key, check free tier limits
    client_ip = request.client.host if request.client else "unknown"

    if client_ip not in USAGE_TRACKER:
        USAGE_TRACKER[client_ip] = {"standard": 0, "agentic": 0}

    usage = USAGE_TRACKER[client_ip]["agentic"]

    if usage >= FREE_TIER_LIMITS["agentic"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Since this is a demo i built for my portfolio, i have set limits ({FREE_TIER_LIMITS['agentic']} requests) to prevent abuse. If i have proided you an API Key, please use it to continue accessing the Agentic RAG endpoints.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Increment usage
    USAGE_TRACKER[client_ip]["agentic"] += 1
    return
