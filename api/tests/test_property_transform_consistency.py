# Feature: vllm-docker-inference, Property 2: İstek/Yanıt Dönüşüm Tutarlılığı
"""Property-based tests for request/response transformation consistency.

**Validates: Requirements 3.4, 10.1, 10.2**

For any valid request object, to_vllm_format must produce a valid vLLM request structure;
and for any valid vLLM response object, from_vllm_format must produce a response conforming
to the API schema.
"""

import time

from hypothesis import given, settings
from hypothesis import strategies as st

from app.models.requests import ChatCompletionRequest, ChatMessage, CompletionRequest
from app.models.responses import ChatCompletionResponse, CompletionResponse
from app.services.transformer import RequestTransformer


# --- Hypothesis Strategies ---

model_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_./"),
    min_size=1,
    max_size=50,
)

non_empty_text = st.text(min_size=1, max_size=200)

role_strategy = st.sampled_from(["system", "user", "assistant"])

chat_message_strategy = st.builds(
    ChatMessage,
    role=role_strategy,
    content=non_empty_text,
)

stop_strategy = st.one_of(
    st.none(),
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=3),
)

completion_request_strategy = st.builds(
    CompletionRequest,
    model=model_name_strategy,
    prompt=non_empty_text,
    max_tokens=st.integers(min_value=1, max_value=4096),
    temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    top_p=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    stream=st.booleans(),
    stop=stop_strategy,
)

chat_completion_request_strategy = st.builds(
    ChatCompletionRequest,
    model=model_name_strategy,
    messages=st.lists(chat_message_strategy, min_size=1, max_size=10),
    max_tokens=st.integers(min_value=1, max_value=4096),
    temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    top_p=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    stream=st.booleans(),
    stop=stop_strategy,
)

# Strategy for vLLM completion response dicts
positive_int = st.integers(min_value=0, max_value=10000)

vllm_completion_response_strategy = st.builds(
    lambda resp_id, model, created, text, finish_reason, pt, ct: {
        "id": resp_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    },
    resp_id=st.text(min_size=1, max_size=50).map(lambda s: f"cmpl-{s}"),
    model=model_name_strategy,
    created=st.integers(min_value=1000000000, max_value=2000000000),
    text=st.text(min_size=0, max_size=200),
    finish_reason=st.sampled_from(["stop", "length", None]),
    pt=positive_int,
    ct=positive_int,
)

# Strategy for vLLM chat completion response dicts
vllm_chat_response_strategy = st.builds(
    lambda resp_id, model, created, content, finish_reason, pt, ct: {
        "id": resp_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    },
    resp_id=st.text(min_size=1, max_size=50).map(lambda s: f"chatcmpl-{s}"),
    model=model_name_strategy,
    created=st.integers(min_value=1000000000, max_value=2000000000),
    content=st.text(min_size=0, max_size=200),
    finish_reason=st.sampled_from(["stop", "length", None]),
    pt=positive_int,
    ct=positive_int,
)


transformer = RequestTransformer()


# --- to_vllm_format output structure tests ---


class TestCompletionToVllmStructure:
    """Property test: to_vllm_format on CompletionRequest always produces a valid vLLM dict."""

    @settings(max_examples=100)
    @given(request=completion_request_strategy)
    def test_to_vllm_format_returns_valid_completion_structure(self, request: CompletionRequest):
        result = transformer.to_vllm_format(request)

        # Must be a dict
        assert isinstance(result, dict)

        # Must contain all required vLLM fields
        assert "model" in result
        assert "prompt" in result
        assert "max_tokens" in result
        assert "temperature" in result
        assert "top_p" in result
        assert "stream" in result

        # Field values must match the input request
        assert result["model"] == request.model
        assert result["prompt"] == request.prompt
        assert result["max_tokens"] == request.max_tokens
        assert result["temperature"] == request.temperature
        assert result["top_p"] == request.top_p
        assert result["stream"] == request.stream

        # stop field: present only when not None
        if request.stop is not None:
            assert result["stop"] == request.stop
        else:
            assert "stop" not in result


class TestChatCompletionToVllmStructure:
    """Property test: to_vllm_format on ChatCompletionRequest always produces a valid vLLM dict."""

    @settings(max_examples=100)
    @given(request=chat_completion_request_strategy)
    def test_to_vllm_format_returns_valid_chat_structure(self, request: ChatCompletionRequest):
        result = transformer.to_vllm_format(request)

        # Must be a dict
        assert isinstance(result, dict)

        # Must contain all required vLLM fields
        assert "model" in result
        assert "messages" in result
        assert "max_tokens" in result
        assert "temperature" in result
        assert "top_p" in result
        assert "stream" in result

        # Field values must match the input request
        assert result["model"] == request.model
        assert result["max_tokens"] == request.max_tokens
        assert result["temperature"] == request.temperature
        assert result["top_p"] == request.top_p
        assert result["stream"] == request.stream

        # Messages must match
        assert len(result["messages"]) == len(request.messages)
        for orig, converted in zip(request.messages, result["messages"]):
            assert isinstance(converted, dict)
            assert converted["role"] == orig.role
            assert converted["content"] == orig.content

        # stop field: present only when not None
        if request.stop is not None:
            assert result["stop"] == request.stop
        else:
            assert "stop" not in result


# --- from_vllm_format output schema tests ---


class TestVllmToCompletionResponseSchema:
    """Property test: from_vllm_format on valid vLLM completion dicts produces a valid CompletionResponse."""

    @settings(max_examples=100)
    @given(vllm_resp=vllm_completion_response_strategy)
    def test_from_vllm_format_returns_valid_completion_response(self, vllm_resp: dict):
        result = transformer.from_vllm_format(vllm_resp, request_type="completion")

        # Must be a valid Pydantic CompletionResponse
        assert isinstance(result, CompletionResponse)

        # Must contain all required fields
        assert result.id == vllm_resp["id"]
        assert result.object is not None
        assert result.created == vllm_resp["created"]
        assert result.model == vllm_resp["model"]
        assert len(result.choices) == len(vllm_resp["choices"])
        assert result.usage is not None
        assert result.usage.prompt_tokens == vllm_resp["usage"]["prompt_tokens"]
        assert result.usage.completion_tokens == vllm_resp["usage"]["completion_tokens"]
        assert result.usage.total_tokens == vllm_resp["usage"]["total_tokens"]


class TestVllmToChatCompletionResponseSchema:
    """Property test: from_vllm_format on valid vLLM chat dicts produces a valid ChatCompletionResponse."""

    @settings(max_examples=100)
    @given(vllm_resp=vllm_chat_response_strategy)
    def test_from_vllm_format_returns_valid_chat_response(self, vllm_resp: dict):
        result = transformer.from_vllm_format(vllm_resp, request_type="chat_completion")

        # Must be a valid Pydantic ChatCompletionResponse
        assert isinstance(result, ChatCompletionResponse)

        # Must contain all required fields
        assert result.id == vllm_resp["id"]
        assert result.object is not None
        assert result.created == vllm_resp["created"]
        assert result.model == vllm_resp["model"]
        assert len(result.choices) == len(vllm_resp["choices"])
        assert result.choices[0].message.role == "assistant"
        assert result.choices[0].message.content == vllm_resp["choices"][0]["message"]["content"]
        assert result.usage is not None
        assert result.usage.prompt_tokens == vllm_resp["usage"]["prompt_tokens"]
        assert result.usage.completion_tokens == vllm_resp["usage"]["completion_tokens"]
        assert result.usage.total_tokens == vllm_resp["usage"]["total_tokens"]
