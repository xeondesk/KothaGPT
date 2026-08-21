from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import Header, HTTPException, status


async def require_api_token(authorization: str | None = Header(default=None)) -> str:
    """Require a bearer token when the API is not explicitly in local mode."""
    configured = os.getenv("KOTHAGPT_API_TOKEN")
    local_mode = os.getenv("KOTHAGPT_LOCAL_MODE", "false").lower() == "true"
    if not configured:
        if local_mode:
            return "local"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="API authentication is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")
    token = authorization.removeprefix("Bearer ").strip()
    if not token or not secrets.compare_digest(token, configured):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")
    return token


ApiToken = Annotated[str, require_api_token]

__all__ = ["ApiToken", "require_api_token"]
