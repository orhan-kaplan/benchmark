"""Shared test fixtures for the vLLM inference API test suite.

Provides a FastAPI TestClient configured with TEST_MODE=true and
AUTH_ENABLED=false so that endpoint tests run without a real vLLM
backend or authentication.
"""

import os
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient


@pytest.fixture()
def test_client():
    """Create a FastAPI TestClient with test-mode environment variables.

    Sets TEST_MODE=true (mock responses) and AUTH_ENABLED=false (no auth)
    then imports the app *inside* the patched environment so that
    module-level config reads pick up the overrides.

    Resets the lazy-initialised module-level registries in every router
    so each test starts with a clean slate.
    """
    env_overrides = {
        "TEST_MODE": "true",
        "AUTH_ENABLED": "false",
        "MODEL_ENDPOINTS": "test-model:http://localhost:9999",
    }

    with patch.dict(os.environ, env_overrides, clear=False):
        # Reset module-level singletons so they re-read env vars
        import app.routers.completions as comp_mod
        import app.routers.chat as chat_mod
        import app.routers.models as models_mod
        import app.routers.health as health_mod

        comp_mod._registry = None
        chat_mod._registry = None
        models_mod._registry = None
        health_mod._registry = None
        health_mod._monitor = None

        # Import app after env is patched
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client

        # Clean up singletons after test
        comp_mod._registry = None
        chat_mod._registry = None
        models_mod._registry = None
        health_mod._registry = None
        health_mod._monitor = None
