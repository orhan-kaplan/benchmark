# Feature: vllm-docker-inference, Property 4: Kimlik Doğrulama Tutarlılığı
"""
Property-based tests for authentication consistency.

**Validates: Requirements 4.1, 4.2, 4.3**

Property 4: For any API key, if the key is in the configured valid keys list
the request must be accepted; if the key is not in the list or the
Authorization header is missing, the system must return HTTP 401.
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.middleware.auth import AuthMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _ok_handler(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def _make_app(api_keys: list[str]) -> Starlette:
    """Create a minimal Starlette app with AuthMiddleware enabled."""
    routes = [Route("/test", _ok_handler)]
    app = Starlette(routes=routes)
    app.add_middleware(AuthMiddleware, auth_enabled=True, api_keys=api_keys)
    return app


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate printable, non-empty, ASCII-safe API key strings
# HTTP headers require ASCII-encodable values, so we restrict to safe chars.
_KEY_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
api_key_st = st.text(alphabet=_KEY_ALPHABET, min_size=1, max_size=64)

# A non-empty set of valid API keys
valid_keys_st = st.lists(api_key_st, min_size=1, max_size=10, unique=True)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestAuthConsistency:
    """Property 4: Kimlik Doğrulama Tutarlılığı."""

    @given(
        valid_keys=valid_keys_st,
        key_index=st.integers(min_value=0, max_value=9),
    )
    @settings(max_examples=100)
    def test_valid_key_accepted(self, valid_keys, key_index):
        """A request bearing a valid API key must be accepted (HTTP 200).

        **Validates: Requirements 4.1, 4.2**
        """
        key = valid_keys[key_index % len(valid_keys)]
        client = TestClient(_make_app(valid_keys), raise_server_exceptions=False)
        resp = client.get("/test", headers={"Authorization": f"Bearer {key}"})
        assert resp.status_code == 200

    @given(
        valid_keys=valid_keys_st,
        random_token=api_key_st,
    )
    @settings(max_examples=100)
    def test_invalid_key_rejected(self, valid_keys, random_token):
        """A request with a key NOT in the valid set must return HTTP 401.

        **Validates: Requirements 4.1, 4.3**
        """
        assume(random_token not in valid_keys)
        client = TestClient(_make_app(valid_keys), raise_server_exceptions=False)
        resp = client.get("/test", headers={"Authorization": f"Bearer {random_token}"})
        assert resp.status_code == 401

    @given(valid_keys=valid_keys_st)
    @settings(max_examples=100)
    def test_missing_header_rejected(self, valid_keys):
        """A request with no Authorization header must return HTTP 401.

        **Validates: Requirements 4.3**
        """
        client = TestClient(_make_app(valid_keys), raise_server_exceptions=False)
        resp = client.get("/test")
        assert resp.status_code == 401
