"""Integration tests for API endpoints using FastAPI TestClient.

Tests run with TEST_MODE=true and AUTH_ENABLED=false so no real vLLM
backend or authentication is required.

Validates: Requirements 9.4, 3.5, 3.6, 4.1, 4.2, 4.3, 4.5, 5.1, 5.2, 5.4
"""


# ------------------------------------------------------------------
# POST /v1/completions
# ------------------------------------------------------------------


class TestCompletionsEndpoint:
    """Tests for the /v1/completions endpoint in test mode."""

    def test_completions_returns_mock_response(self, test_client):
        """A valid completion request returns 200 with mock data."""
        payload = {"model": "test-model", "prompt": "Hello"}
        resp = test_client.post("/v1/completions", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "text_completion"
        assert body["model"] == "test-model"
        assert len(body["choices"]) >= 1
        assert "text" in body["choices"][0]
        assert "usage" in body

    def test_completions_with_all_optional_fields(self, test_client):
        """Optional fields are accepted without error."""
        payload = {
            "model": "test-model",
            "prompt": "Translate:",
            "max_tokens": 100,
            "temperature": 0.5,
            "top_p": 0.9,
            "stream": False,
            "stop": ["\n"],
        }
        resp = test_client.post("/v1/completions", json=payload)
        assert resp.status_code == 200

    def test_completions_missing_model_returns_422(self, test_client):
        """Missing required 'model' field returns 422."""
        payload = {"prompt": "Hello"}
        resp = test_client.post("/v1/completions", json=payload)
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["type"] == "validation_error"

    def test_completions_missing_prompt_returns_422(self, test_client):
        """Missing required 'prompt' field returns 422."""
        payload = {"model": "test-model"}
        resp = test_client.post("/v1/completions", json=payload)
        assert resp.status_code == 422

    def test_completions_empty_body_returns_422(self, test_client):
        """Empty JSON body returns 422."""
        resp = test_client.post("/v1/completions", json={})
        assert resp.status_code == 422

    def test_completions_invalid_type_returns_422(self, test_client):
        """Wrong type for a field returns 422."""
        payload = {"model": "test-model", "prompt": "Hi", "max_tokens": "not_a_number"}
        resp = test_client.post("/v1/completions", json=payload)
        assert resp.status_code == 422


# ------------------------------------------------------------------
# POST /v1/chat/completions
# ------------------------------------------------------------------


class TestChatCompletionsEndpoint:
    """Tests for the /v1/chat/completions endpoint in test mode."""

    def test_chat_completions_returns_mock_response(self, test_client):
        """A valid chat request returns 200 with mock data."""
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        resp = test_client.post("/v1/chat/completions", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "test-model"
        assert len(body["choices"]) >= 1
        assert "message" in body["choices"][0]
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in body

    def test_chat_completions_with_multiple_messages(self, test_client):
        """Multiple messages in the conversation are accepted."""
        payload = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        resp = test_client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200

    def test_chat_completions_missing_model_returns_422(self, test_client):
        """Missing 'model' field returns 422."""
        payload = {"messages": [{"role": "user", "content": "Hi"}]}
        resp = test_client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422

    def test_chat_completions_missing_messages_returns_422(self, test_client):
        """Missing 'messages' field returns 422."""
        payload = {"model": "test-model"}
        resp = test_client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422

    def test_chat_completions_empty_body_returns_422(self, test_client):
        """Empty JSON body returns 422."""
        resp = test_client.post("/v1/chat/completions", json={})
        assert resp.status_code == 422

    def test_chat_completions_invalid_role_returns_422(self, test_client):
        """Invalid message role returns 422."""
        payload = {
            "model": "test-model",
            "messages": [{"role": "invalid_role", "content": "Hi"}],
        }
        resp = test_client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422


# ------------------------------------------------------------------
# GET /v1/models
# ------------------------------------------------------------------


class TestModelsEndpoint:
    """Tests for the /v1/models endpoint."""

    def test_models_returns_list(self, test_client):
        """GET /v1/models returns a model list."""
        resp = test_client.get("/v1/models")

        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert isinstance(body["data"], list)
        assert len(body["data"]) >= 1
        assert body["data"][0]["id"] == "test-model"
        assert body["data"][0]["object"] == "model"


# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_status(self, test_client):
        """GET /health returns a health response with status and engines."""
        resp = test_client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert body["status"] in ("healthy", "degraded", "unhealthy")
        assert "engines" in body
        assert "timestamp" in body


# ------------------------------------------------------------------
# GET /metrics
# ------------------------------------------------------------------


class TestMetricsEndpoint:
    """Tests for the /metrics endpoint."""

    def test_metrics_returns_system_metrics(self, test_client):
        """GET /metrics returns system metrics."""
        resp = test_client.get("/metrics")

        assert resp.status_code == 200
        body = resp.json()
        assert "active_requests" in body
        assert "uptime_seconds" in body
        assert isinstance(body["uptime_seconds"], (int, float))


# ------------------------------------------------------------------
# GET /docs
# ------------------------------------------------------------------


class TestDocsEndpoint:
    """Tests for the /docs (OpenAPI/Swagger) endpoint."""

    def test_docs_returns_html(self, test_client):
        """GET /docs returns the Swagger UI page."""
        resp = test_client.get("/docs")

        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
