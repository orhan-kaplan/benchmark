"""Rate limiting middleware using token bucket algorithm.

Implements per-API-key rate limiting with configurable requests per minute.
Adds X-RateLimit-Remaining header to responses and returns HTTP 429
with Retry-After header when the limit is exceeded.
"""

import math
import time
from typing import Dict, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Paths that bypass rate limiting
_PUBLIC_PATHS: Set[str] = {"/health", "/docs", "/openapi.json", "/metrics"}


class _TokenBucket:
    """Token bucket for a single API key.

    Attributes:
        tokens: Current number of available tokens.
        last_refill: Timestamp of the last refill.
        max_tokens: Maximum bucket capacity (= requests_per_minute).
        refill_rate: Tokens added per second (= requests_per_minute / 60).
    """

    __slots__ = ("tokens", "last_refill", "max_tokens", "refill_rate")

    def __init__(self, max_tokens: int, refill_rate: float, now: float | None = None) -> None:
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = float(max_tokens)
        self.last_refill = now if now is not None else time.monotonic()

    def refill(self, now: float | None = None) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = now if now is not None else time.monotonic()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

    def consume(self, now: float | None = None) -> bool:
        """Try to consume one token. Returns True if successful."""
        self.refill(now)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def seconds_until_next_token(self) -> float:
        """Seconds until at least one token is available."""
        if self.tokens >= 1.0:
            return 0.0
        deficit = 1.0 - self.tokens
        return math.ceil(deficit / self.refill_rate) if self.refill_rate > 0 else 1.0


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Per-API-key rate limiting middleware using token bucket algorithm.

    Args:
        app: The ASGI application.
        requests_per_minute: Maximum requests allowed per minute per API key.
    """

    def __init__(self, app, requests_per_minute: int = 60) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.refill_rate = requests_per_minute / 60.0
        self._buckets: Dict[str, _TokenBucket] = {}

    def _get_bucket(self, key: str) -> _TokenBucket:
        """Get or create a token bucket for the given API key."""
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _TokenBucket(
                max_tokens=self.requests_per_minute,
                refill_rate=self.refill_rate,
            )
            self._buckets[key] = bucket
        return bucket

    @staticmethod
    def _extract_api_key(request: Request) -> str:
        """Extract API key from Authorization: Bearer <token> header."""
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        return "anonymous"

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for public paths
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        api_key = self._extract_api_key(request)
        bucket = self._get_bucket(api_key)

        if bucket.consume():
            response = await call_next(request)
            response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
            return response

        # Rate limit exceeded
        retry_after = bucket.seconds_until_next_token()
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "type": "rate_limit_error",
                    "message": "Rate limit exceeded. Please retry after the indicated time.",
                }
            },
            headers={
                "Retry-After": str(int(retry_after)),
                "X-RateLimit-Remaining": "0",
            },
        )
