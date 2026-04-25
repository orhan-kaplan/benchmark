# Feature: vllm-docker-inference, Property 5: Hız Sınırlama İnvariantı
"""
Property-based tests for rate limiting invariant.

**Validates: Requirements 5.1, 5.2, 5.4**

Property 5: For any API key and request sequence, the token bucket algorithm
must preserve these invariants:
  (a) accepted requests per minute never exceed the configured limit,
  (b) every response carries a correct X-RateLimit-Remaining header,
  (c) when the limit is exceeded HTTP 429 is returned with Retry-After header.
"""

from hypothesis import given, settings
from hypothesis import strategies as st
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.middleware.rate_limiter import RateLimiterMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _ok_handler(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def _make_app(rpm: int) -> Starlette:
    """Create a minimal Starlette app with RateLimiterMiddleware."""
    app = Starlette(routes=[Route("/", _ok_handler)])
    app.add_middleware(RateLimiterMiddleware, requests_per_minute=rpm)
    return app


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# RPM between 1 and 20 to keep tests fast
rpm_st = st.integers(min_value=1, max_value=20)

# Number of requests to send: at least 1, up to 40 (enough to exceed any RPM ≤ 20)
num_requests_st = st.integers(min_value=1, max_value=40)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

class TestRateLimitInvariant:
    """Property 5: Hız Sınırlama İnvariantı."""

    @given(rpm=rpm_st, num_requests=num_requests_st)
    @settings(max_examples=100)
    def test_accepted_requests_never_exceed_limit(self, rpm, num_requests):
        """The number of accepted (200) requests must never exceed the RPM limit.

        **Validates: Requirements 5.1**
        """
        client = TestClient(_make_app(rpm), raise_server_exceptions=False)
        accepted = 0
        for _ in range(num_requests):
            resp = client.get("/", headers={"Authorization": "Bearer prop-key"})
            if resp.status_code == 200:
                accepted += 1
        assert accepted <= rpm

    @given(rpm=rpm_st, num_requests=num_requests_st)
    @settings(max_examples=100)
    def test_remaining_header_present_and_non_negative(self, rpm, num_requests):
        """Every 200 response must include X-RateLimit-Remaining with a non-negative value.

        **Validates: Requirements 5.4**
        """
        client = TestClient(_make_app(rpm), raise_server_exceptions=False)
        for _ in range(num_requests):
            resp = client.get("/", headers={"Authorization": "Bearer hdr-key"})
            if resp.status_code == 200:
                assert "X-RateLimit-Remaining" in resp.headers, (
                    "200 response missing X-RateLimit-Remaining header"
                )
                remaining = int(resp.headers["X-RateLimit-Remaining"])
                assert remaining >= 0, (
                    f"X-RateLimit-Remaining must be non-negative, got {remaining}"
                )

    @given(rpm=rpm_st, num_requests=num_requests_st)
    @settings(max_examples=100)
    def test_429_has_retry_after_and_remaining_zero(self, rpm, num_requests):
        """When 429 is returned, Retry-After must be present and X-RateLimit-Remaining must be '0'.

        **Validates: Requirements 5.2, 5.4**
        """
        client = TestClient(_make_app(rpm), raise_server_exceptions=False)
        for _ in range(num_requests):
            resp = client.get("/", headers={"Authorization": "Bearer limit-key"})
            if resp.status_code == 429:
                assert "Retry-After" in resp.headers, (
                    "429 response missing Retry-After header"
                )
                assert resp.headers["X-RateLimit-Remaining"] == "0", (
                    f"Expected X-RateLimit-Remaining='0' on 429, "
                    f"got '{resp.headers.get('X-RateLimit-Remaining')}'"
                )
