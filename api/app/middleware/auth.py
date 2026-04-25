"""Authentication middleware for Bearer token validation.

Validates Authorization header with Bearer token against configured API keys.
Can be disabled via AUTH_ENABLED=false configuration.
"""

import json
from typing import List

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Paths that bypass authentication
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/metrics"}


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that validates Bearer tokens in the Authorization header.

    Args:
        app: The ASGI application.
        auth_enabled: Whether authentication is active.
        api_keys: List of valid API keys (Bearer tokens).
    """

    def __init__(self, app, auth_enabled: bool = True, api_keys: List[str] | None = None):
        super().__init__(app)
        self.auth_enabled = auth_enabled
        self.api_keys = api_keys or []

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip auth when disabled
        if not self.auth_enabled:
            return await call_next(request)

        # Skip auth for public endpoints
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Extract Authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header:
            return _unauthorized("Missing Authorization header")

        if not auth_header.startswith("Bearer "):
            return _unauthorized("Invalid Authorization header format. Expected: Bearer <token>")

        token = auth_header[7:]  # Strip "Bearer " prefix

        if token not in self.api_keys:
            return _unauthorized("Invalid API key")

        return await call_next(request)


def _unauthorized(message: str) -> JSONResponse:
    """Return a 401 JSON response in ErrorResponse format."""
    return JSONResponse(
        status_code=401,
        content={"error": {"type": "authentication_error", "message": message}},
    )
