"""APIClient birim testleri."""

import json

import httpx
import pytest

from benchmark.api_client import APIClient
from benchmark.models import GenerationParams


@pytest.fixture
def api_client():
    """Varsayılan APIClient instance'ı."""
    return APIClient(timeout=5.0, max_retries=2)


@pytest.fixture
def default_params():
    return GenerationParams(temperature=0.7, max_tokens=256, top_p=1.0)


@pytest.fixture
def default_messages():
    return [{"role": "user", "content": "Hello"}]


class TestAPIClientInit:
    def test_default_values(self):
        client = APIClient()
        assert client.timeout == 120.0
        assert client.max_retries == 3

    def test_custom_values(self):
        client = APIClient(timeout=30.0, max_retries=5)
        assert client.timeout == 30.0
        assert client.max_retries == 5

    def test_httpx_client_created(self):
        client = APIClient(timeout=10.0)
        assert isinstance(client._client, httpx.AsyncClient)


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_success(self, api_client, httpx_mock):
        httpx_mock.add_response(url="http://localhost:8000", status_code=200)
        result = await api_client.health_check("http://localhost:8000")
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure_status(self, api_client, httpx_mock):
        httpx_mock.add_response(url="http://localhost:8000", status_code=500)
        result = await api_client.health_check("http://localhost:8000")
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, api_client, httpx_mock):
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        result = await api_client.health_check("http://localhost:8000")
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_trailing_slash(self, api_client, httpx_mock):
        httpx_mock.add_response(url="http://localhost:8000", status_code=200)
        result = await api_client.health_check("http://localhost:8000/")
        assert result is True


class TestChatCompletion:
    @pytest.mark.asyncio
    async def test_successful_streaming_response(
        self, api_client, default_params, default_messages, httpx_mock
    ):
        """Başarılı streaming yanıt testi."""
        sse_body = (
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"!"}}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}\n\n'
            "data: [DONE]\n\n"
        )
        httpx_mock.add_response(
            url="http://localhost:8000/v1/chat/completions",
            status_code=200,
            text=sse_body,
        )

        result = await api_client.chat_completion(
            "http://localhost:8000", "test-model", default_messages, default_params
        )

        assert result.status_code == 200
        assert result.response_text == "Hello world!"
        assert result.error is None
        assert result.elapsed_ms > 0
        assert result.usage == {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }

    @pytest.mark.asyncio
    async def test_http_error_response(
        self, api_client, default_params, default_messages, httpx_mock
    ):
        """HTTP hata kodu testi."""
        httpx_mock.add_response(
            url="http://localhost:8000/v1/chat/completions",
            status_code=500,
            text="Internal Server Error",
        )

        result = await api_client.chat_completion(
            "http://localhost:8000", "test-model", default_messages, default_params
        )

        assert result.status_code == 500
        assert result.error is not None
        assert "500" in result.error
        assert result.response_text is None

    @pytest.mark.asyncio
    async def test_retry_on_network_error(
        self, default_params, default_messages, httpx_mock
    ):
        """Ağ hatası sonrası retry testi."""
        client = APIClient(timeout=5.0, max_retries=2)

        # İlk deneme başarısız, ikinci başarılı
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        sse_body = (
            'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        httpx_mock.add_response(
            url="http://localhost:8000/v1/chat/completions",
            status_code=200,
            text=sse_body,
        )

        result = await client.chat_completion(
            "http://localhost:8000", "test-model", default_messages, default_params
        )

        assert result.status_code == 200
        assert result.response_text == "OK"

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(
        self, default_params, default_messages, httpx_mock
    ):
        """Tüm retry denemeleri tükendiğinde hata döner."""
        client = APIClient(timeout=5.0, max_retries=2)

        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))

        result = await client.chat_completion(
            "http://localhost:8000", "test-model", default_messages, default_params
        )

        assert result.status_code == 0
        assert result.error is not None
        assert "başarısız" in result.error
        assert result.response_text is None

    @pytest.mark.asyncio
    async def test_ngrok_url_support(
        self, api_client, default_params, default_messages, httpx_mock
    ):
        """ngrok URL desteği testi."""
        sse_body = (
            'data: {"choices":[{"delta":{"content":"ngrok works"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        httpx_mock.add_response(
            url="https://abc123.ngrok-free.app/v1/chat/completions",
            status_code=200,
            text=sse_body,
        )

        result = await api_client.chat_completion(
            "https://abc123.ngrok-free.app",
            "test-model",
            default_messages,
            default_params,
        )

        assert result.status_code == 200
        assert result.response_text == "ngrok works"

    @pytest.mark.asyncio
    async def test_endpoint_trailing_slash(
        self, api_client, default_params, default_messages, httpx_mock
    ):
        """Endpoint sonundaki slash temizlenir."""
        sse_body = (
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        httpx_mock.add_response(
            url="http://localhost:8000/v1/chat/completions",
            status_code=200,
            text=sse_body,
        )

        result = await api_client.chat_completion(
            "http://localhost:8000/",
            "test-model",
            default_messages,
            default_params,
        )

        assert result.status_code == 200
        assert result.response_text == "ok"

    @pytest.mark.asyncio
    async def test_empty_streaming_response(
        self, api_client, default_params, default_messages, httpx_mock
    ):
        """Boş streaming yanıt testi."""
        sse_body = "data: [DONE]\n\n"
        httpx_mock.add_response(
            url="http://localhost:8000/v1/chat/completions",
            status_code=200,
            text=sse_body,
        )

        result = await api_client.chat_completion(
            "http://localhost:8000", "test-model", default_messages, default_params
        )

        assert result.status_code == 200
        assert result.response_text is None
        assert result.error is None

    @pytest.mark.asyncio
    async def test_request_payload_format(
        self, api_client, default_messages, httpx_mock
    ):
        """İstek payload'ının OpenAI uyumlu formatında olduğunu doğrula."""
        params = GenerationParams(temperature=0.5, max_tokens=512, top_p=0.9)
        sse_body = "data: [DONE]\n\n"
        httpx_mock.add_response(
            url="http://localhost:8000/v1/chat/completions",
            status_code=200,
            text=sse_body,
        )

        await api_client.chat_completion(
            "http://localhost:8000", "my-model", default_messages, params
        )

        request = httpx_mock.get_request()
        body = json.loads(request.content)
        assert body["model"] == "my-model"
        assert body["messages"] == default_messages
        assert body["temperature"] == 0.5
        assert body["max_tokens"] == 512
        assert body["top_p"] == 0.9
        assert body["stream"] is True


class TestClose:
    @pytest.mark.asyncio
    async def test_close(self):
        client = APIClient()
        await client.close()
        assert client._client.is_closed
