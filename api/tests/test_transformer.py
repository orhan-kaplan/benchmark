"""Unit tests for the RequestTransformer service."""

import time

import pytest

from app.models.requests import ChatCompletionRequest, ChatMessage, CompletionRequest
from app.models.responses import (
    ChatCompletionResponse,
    CompletionResponse,
)
from app.services.transformer import RequestTransformer, TransformationError


@pytest.fixture
def transformer():
    return RequestTransformer()


# --- to_vllm_format: CompletionRequest ---


class TestCompletionToVllm:
    def test_basic_completion_request(self, transformer):
        req = CompletionRequest(model="gpt-2", prompt="Hello world")
        result = transformer.to_vllm_format(req)

        assert result["model"] == "gpt-2"
        assert result["prompt"] == "Hello world"
        assert result["max_tokens"] == 256
        assert result["temperature"] == 1.0
        assert result["top_p"] == 1.0
        assert result["stream"] is False
        assert "stop" not in result

    def test_completion_with_all_fields(self, transformer):
        req = CompletionRequest(
            model="llama-7b",
            prompt="Translate:",
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            stream=True,
            stop=["\n", "END"],
        )
        result = transformer.to_vllm_format(req)

        assert result["model"] == "llama-7b"
        assert result["prompt"] == "Translate:"
        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.9
        assert result["stream"] is True
        assert result["stop"] == ["\n", "END"]

    def test_completion_stop_none_excluded(self, transformer):
        req = CompletionRequest(model="m", prompt="p", stop=None)
        result = transformer.to_vllm_format(req)
        assert "stop" not in result


# --- to_vllm_format: ChatCompletionRequest ---


class TestChatCompletionToVllm:
    def test_basic_chat_request(self, transformer):
        req = ChatCompletionRequest(
            model="gpt-2",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        result = transformer.to_vllm_format(req)

        assert result["model"] == "gpt-2"
        assert result["messages"] == [{"role": "user", "content": "Hi"}]
        assert result["max_tokens"] == 256
        assert "stop" not in result

    def test_chat_with_multiple_messages(self, transformer):
        req = ChatCompletionRequest(
            model="llama",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi there!"),
            ],
            stop=["STOP"],
        )
        result = transformer.to_vllm_format(req)

        assert len(result["messages"]) == 3
        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert result["messages"][2] == {"role": "assistant", "content": "Hi there!"}
        assert result["stop"] == ["STOP"]


# --- to_vllm_format: unsupported type ---


class TestToVllmUnsupported:
    def test_unsupported_type_raises(self, transformer):
        with pytest.raises(TransformationError, match="Unsupported request type"):
            transformer.to_vllm_format("not a request")  # type: ignore


# --- from_vllm_format: completion ---


class TestVllmToCompletion:
    def _make_vllm_completion_response(self, **overrides):
        base = {
            "id": "cmpl-123",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "gpt-2",
            "choices": [
                {"index": 0, "text": "Hello!", "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        }
        base.update(overrides)
        return base

    def test_basic_completion_response(self, transformer):
        vllm_resp = self._make_vllm_completion_response()
        result = transformer.from_vllm_format(vllm_resp, request_type="completion")

        assert isinstance(result, CompletionResponse)
        assert result.id == "cmpl-123"
        assert result.model == "gpt-2"
        assert len(result.choices) == 1
        assert result.choices[0].text == "Hello!"
        assert result.usage.total_tokens == 8

    def test_missing_required_field_raises(self, transformer):
        with pytest.raises(TransformationError, match="Missing required fields"):
            transformer.from_vllm_format({"id": "x"}, request_type="completion")

    def test_non_dict_raises(self, transformer):
        with pytest.raises(TransformationError):
            transformer.from_vllm_format("not a dict", request_type="completion")  # type: ignore


# --- from_vllm_format: chat_completion ---


class TestVllmToChatCompletion:
    def _make_vllm_chat_response(self, **overrides):
        base = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "llama",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        }
        base.update(overrides)
        return base

    def test_basic_chat_response(self, transformer):
        vllm_resp = self._make_vllm_chat_response()
        result = transformer.from_vllm_format(vllm_resp, request_type="chat_completion")

        assert isinstance(result, ChatCompletionResponse)
        assert result.id == "chatcmpl-456"
        assert result.choices[0].message.role == "assistant"
        assert result.choices[0].message.content == "Hi!"
        assert result.usage.total_tokens == 12

    def test_missing_fields_raises(self, transformer):
        with pytest.raises(TransformationError, match="Missing required fields"):
            transformer.from_vllm_format({}, request_type="chat_completion")


# --- from_vllm_format: unsupported type ---


class TestFromVllmUnsupported:
    def test_unsupported_request_type_raises(self, transformer):
        with pytest.raises(TransformationError, match="Unsupported request_type"):
            transformer.from_vllm_format({}, request_type="unknown")
