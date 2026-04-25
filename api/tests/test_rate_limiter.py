"""Unit tests for the RateLimiterMiddleware."""

import time
from unittest.mock import patch

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.middleware.rate_limiter import RateLimiterMiddleware, _TokenBucket


# ---------------------------------------------------------------------------
# _TokenBucket unit tests
# ---------------------------------------------------------------------------

class TestTokenBucket:
    def test_initial_tokens_equal_max(self):
        bucket = _TokenBucket(max_tokens=10, refill_rate=10 / 60, now=0.0)
        assert bucket.tokens == 10.0

    def test_consume_decrements_token(self):
        bucket = _TokenBucket(max_tokens=10, refill_rate=10 / 60, now=0.0)
        assert bucket.consume(now=0.0) is True
        assert bucket.tokens == 9.0

    def test_consume_fails_when_empty(self):
        bucket = _TokenBucket(max_tokens=1, refill_rate=1 / 60, now=0.0)
        assert bucket.consume(now=0.0) is True
        assert bucket.consume(now=0.0) is False

    def test_refill_adds_tokens_over_time(self):
        bucket = _TokenBucket(max_tokens=60, refill_rate=1.0, now=0.0)
        # Consume all
        for _ in range(60):
            bucket.consume(now=0.0)
        assert bucket.tokens < 1.0
        # Refill after 5 seconds → 5 tokens
        bucket.refill(now=5.0)
        assert 4.9 <= bucket.tokens <= 5.1

    def test_refill_caps_at_max(self):
        bucket = _TokenBucket(max_tokens=10, refill_rate=10 / 60, now=0.0)
        bucket.refill(now=1000.0)
        assert bucket.tokens == 10.0

    def test_seconds_until_next_token(self):
        bucket = _TokenBucket(max_tokens=60, refill_rate=1.0, now=0.0)
        for _ in range(60):
            bucket.consume(now=0.0)
        wait = bucket.seconds_until_next_token()
        assert wait >= 1


# ---------------------------------------------------------------------------
# Middleware integration tests
# ---------------------------------------------------------------------------

def _make_app(rpm: int = 5) -> Starlette:
    """Create a minimal Starlette app with rate limiter middleware."""

    async def homepage(request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    async def health(request: Request) -> PlainTextResponse:
        return PlainTextResponse("healthy")

    app = Starlette(
        routes=[
            Route("/", homepage),
            Route("/health", health),
        ],
    )
    app.add_middleware(RateLimiterMiddleware, requests_per_minute=rpm)
    return app


class TestRateLimiterMiddleware:
    def test_allows_requests_within_limit(self):
        client = TestClient(_make_app(rpm=5))
        resp = client.get("/", headers={"Authorization": "Bearer test-key"})
        assert resp.status_code == 200
        assert "X-RateLimit-Remaining" in resp.headers

    def test_returns_429_when_limit_exceeded(self):
        client = TestClient(_make_app(rpm=3))
        for _ in range(3):
            resp = client.get("/", headers={"Authorization": "Bearer key1"})
            assert resp.status_code == 200

        resp = client.get("/", headers={"Authorization": "Bearer key1"})
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        body = resp.json()
        assert body["error"]["type"] == "rate_limit_error"

    def test_different_keys_have_separate_buckets(self):
        client = TestClient(_make_app(rpm=2))
        # Exhaust key-a
        for _ in range(2):
            client.get("/", headers={"Authorization": "Bearer key-a"})
        resp_a = client.get("/", headers={"Authorization": "Bearer key-a"})
        assert resp_a.status_code == 429

        # key-b should still work
        resp_b = client.get("/", headers={"Authorization": "Bearer key-b"})
        assert resp_b.status_code == 200

    def test_public_paths_bypass_rate_limiting(self):
        client = TestClient(_make_app(rpm=1))
        # Exhaust the limit
        client.get("/", headers={"Authorization": "Bearer key"})
        resp = client.get("/", headers={"Authorization": "Bearer key"})
        assert resp.status_code == 429

        # /health should still work
        for _ in range(5):
            resp = client.get("/health", headers={"Authorization": "Bearer key"})
            assert resp.status_code == 200

    def test_anonymous_key_when_no_auth_header(self):
        client = TestClient(_make_app(rpm=2))
        for _ in range(2):
            resp = client.get("/")
            assert resp.status_code == 200
        resp = client.get("/")
        assert resp.status_code == 429

    def test_remaining_header_decreases(self):
        client = TestClient(_make_app(rpm=5))
        resp1 = client.get("/", headers={"Authorization": "Bearer hdr-key"})
        remaining1 = int(resp1.headers["X-RateLimit-Remaining"])

        resp2 = client.get("/", headers={"Authorization": "Bearer hdr-key"})
        remaining2 = int(resp2.headers["X-RateLimit-Remaining"])

        assert remaining2 < remaining1

    def test_429_response_has_retry_after_and_remaining_zero(self):
        client = TestClient(_make_app(rpm=1))
        client.get("/", headers={"Authorization": "Bearer once"})
        resp = client.get("/", headers={"Authorization": "Bearer once"})
        assert resp.status_code == 429
        assert resp.headers["X-RateLimit-Remaining"] == "0"
        assert int(resp.headers["Retry-After"]) >= 0
