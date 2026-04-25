"""Unit tests for the AuthMiddleware."""

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.middleware.auth import AuthMiddleware


# Simple handler for testing
async def _ok_handler(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def _health_handler(request: Request) -> JSONResponse:
    return JSONResponse({"status": "healthy"})


def _make_app(auth_enabled: bool = True, api_keys: list | None = None):
    """Create a minimal Starlette app with AuthMiddleware."""
    routes = [
        Route("/test", _ok_handler),
        Route("/health", _health_handler),
        Route("/docs", _ok_handler),
        Route("/openapi.json", _ok_handler),
        Route("/metrics", _ok_handler),
    ]
    app = Starlette(routes=routes)
    app.add_middleware(AuthMiddleware, auth_enabled=auth_enabled, api_keys=api_keys)
    return app


class TestAuthMiddlewareEnabled:
    """Tests when auth is enabled."""

    def setup_method(self):
        self.app = _make_app(auth_enabled=True, api_keys=["valid-key-1", "valid-key-2"])
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_valid_token_passes(self):
        resp = self.client.get("/test", headers={"Authorization": "Bearer valid-key-1"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_second_valid_token_passes(self):
        resp = self.client.get("/test", headers={"Authorization": "Bearer valid-key-2"})
        assert resp.status_code == 200

    def test_missing_header_returns_401(self):
        resp = self.client.get("/test")
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["type"] == "authentication_error"
        assert "Missing" in body["error"]["message"]

    def test_invalid_token_returns_401(self):
        resp = self.client.get("/test", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["type"] == "authentication_error"
        assert "Invalid API key" in body["error"]["message"]

    def test_malformed_header_no_bearer_prefix(self):
        resp = self.client.get("/test", headers={"Authorization": "Token abc"})
        assert resp.status_code == 401
        body = resp.json()
        assert "Bearer" in body["error"]["message"]

    def test_empty_bearer_token(self):
        resp = self.client.get("/test", headers={"Authorization": "Bearer "})
        assert resp.status_code == 401

    def test_health_endpoint_bypasses_auth(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200

    def test_docs_endpoint_bypasses_auth(self):
        resp = self.client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_endpoint_bypasses_auth(self):
        resp = self.client.get("/openapi.json")
        assert resp.status_code == 200

    def test_metrics_endpoint_bypasses_auth(self):
        resp = self.client.get("/metrics")
        assert resp.status_code == 200


class TestAuthMiddlewareDisabled:
    """Tests when auth is disabled (AUTH_ENABLED=false)."""

    def setup_method(self):
        self.app = _make_app(auth_enabled=False, api_keys=["some-key"])
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_no_header_passes_when_disabled(self):
        resp = self.client.get("/test")
        assert resp.status_code == 200

    def test_invalid_token_passes_when_disabled(self):
        resp = self.client.get("/test", headers={"Authorization": "Bearer bad"})
        assert resp.status_code == 200


class TestAuthMiddlewareEmptyKeys:
    """Tests when auth is enabled but no keys are configured."""

    def setup_method(self):
        self.app = _make_app(auth_enabled=True, api_keys=[])
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_any_token_rejected_with_empty_keys(self):
        resp = self.client.get("/test", headers={"Authorization": "Bearer anything"})
        assert resp.status_code == 401

    def test_public_paths_still_accessible(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
