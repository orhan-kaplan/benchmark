"""FastAPI main application for vLLM Inference API.

Creates the FastAPI app, registers middleware (Auth → Rate Limiter → Logging),
connects all routers, configures OpenAPI/Swagger, and defines custom exception
handlers for consistent ErrorResponse formatting.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import get_config
from app.middleware.auth import AuthMiddleware
from app.middleware.logging import RequestLoggingMiddleware
from app.middleware.rate_limiter import RateLimiterMiddleware
from app.models.responses import ErrorDetail, ErrorResponse
from app.routers import chat, completions
from app.routers import health as health_router
from app.routers import models as models_router

logger = logging.getLogger(__name__)

config = get_config()

app = FastAPI(
    title="vLLM Inference API",
    version="1.0.0",
    docs_url="/docs",
)

# ---------------------------------------------------------------------------
# Middleware registration
# Starlette processes middleware in REVERSE order of registration, so we
# register: Logging → Rate Limiter → Auth  ⇒  request hits Auth first.
# ---------------------------------------------------------------------------
app.add_middleware(RequestLoggingMiddleware, log_level=config.log_level)
app.add_middleware(
    RateLimiterMiddleware, requests_per_minute=config.rate_limit.requests_per_minute
)
app.add_middleware(
    AuthMiddleware, auth_enabled=config.auth_enabled, api_keys=config.api_keys
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(completions.router)
app.include_router(chat.router)
app.include_router(models_router.router)
app.include_router(health_router.router)


# ---------------------------------------------------------------------------
# Custom exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 422 with ErrorResponse format for Pydantic validation failures."""
    body = ErrorResponse(
        error=ErrorDetail(
            type="validation_error",
            message=str(exc.errors()),
        )
    )
    return JSONResponse(status_code=422, content=body.model_dump())


@app.exception_handler(Exception)
async def generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Return 500 with ErrorResponse format for unhandled exceptions."""
    logger.exception("Unhandled exception: %s", exc)
    body = ErrorResponse(
        error=ErrorDetail(
            type="internal_error",
            message="An unexpected error occurred.",
        )
    )
    return JSONResponse(status_code=500, content=body.model_dump())
