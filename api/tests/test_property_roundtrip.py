# Feature: vllm-docker-inference, Property 1: İstek Serileştirme Round-Trip
"""Property-based tests for request serialization round-trip.

**Validates: Requirements 10.3**

For any valid CompletionRequest or ChatCompletionRequest, converting to vLLM format
(to_vllm_format) and then converting a simulated vLLM response back (from_vllm_format)
should produce a valid response with matching model name.
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
    stream=st.just(False),
    stop=stop_strategy,
)

chat_completion_request_strategy = st.builds(
    ChatCompletionRequest,
    model=model_name_strategy,
    messages=st.lists(chat_message_strategy, min_size=1, max_size=10),
    max_tokens=st.integers(min_value=1, max_value=4096),
    temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    top_p=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    stream=st.just(False),
    stop=stop_strategy,
)


# --- Round-Trip Tests ---

transformer = RequestTransformer()


class TestCompletionRoundTrip:
    """Round-trip property test for CompletionRequest."""

    @settings(max_examples=100)
    @given(request=completion_request_strategy)
    def test_completion_roundtrip_produces_valid_response(self, request: CompletionRequest):
        """Converting a CompletionRequest to vLLM format and back yields a valid CompletionResponse
        with matching model name."""
        vllm_dict = transformer.to_vllm_format(request)

        # Verify the vLLM dict preserves the model name
        assert vllm_dict["model"] == request.model
        assert vllm_dict["prompt"] == request.prompt

        # Simulate a vLLM response based on the converted request
        vllm_response = {
            "id": "cmpl-test-001",
            "object": "text_completion",
            "created": int(time.time()),
            "model": vllm_dict["model"],
            "choices": [
                {
                    "index": 0,
                    "text": "generated text",
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        }

        result = transformer.from_vllm_format(vllm_response, request_type="completion")

        # The round-trip must produce a valid CompletionResponse with matching model
        assert isinstance(result, CompletionResponse)
        assert result.model == request.model
        assert result.id == "cmpl-test-001"
        assert result.object == "text_completion"
        assert len(result.choices) == 1
        assert result.usage.total_tokens == 8


class TestChatCompletionRoundTrip:
    """Round-trip property test for ChatCompletionRequest."""

    @settings(max_examples=100)
    @given(request=chat_completion_request_strategy)
    def test_chat_completion_roundtrip_produces_valid_response(
        self, request: ChatCompletionRequest
    ):
        """Converting a ChatCompletionRequest to vLLM format and back yields a valid
        ChatCompletionResponse with matching model and preserved message structure."""
        vllm_dict = transformer.to_vllm_format(request)

        # Verify the vLLM dict preserves model and messages
        assert vllm_dict["model"] == request.model
        assert len(vllm_dict["messages"]) == len(request.messages)
        for orig_msg, vllm_msg in zip(request.messages, vllm_dict["messages"]):
            assert vllm_msg["role"] == orig_msg.role
            assert vllm_msg["content"] == orig_msg.content

        # Simulate a vLLM chat response
        vllm_response = {
            "id": "chatcmpl-test-001",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": vllm_dict["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "response text"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        result = transformer.from_vllm_format(vllm_response, request_type="chat_completion")

        # The round-trip must produce a valid ChatCompletionResponse with matching model
        assert isinstance(result, ChatCompletionResponse)
        assert result.model == request.model
        assert result.id == "chatcmpl-test-001"
        assert result.object == "chat.completion"
        assert len(result.choices) == 1
        assert result.choices[0].message.role == "assistant"
        assert result.usage.total_tokens == 15
