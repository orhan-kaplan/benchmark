# Feature: vllm-docker-inference, Property 3: Pydantic Doğrulama Tutarlılığı
"""
Property-based tests for Pydantic model validation consistency.

**Validates: Requirements 3.5, 3.6, 10.5**

Property 3: For any valid request object, Pydantic validation must succeed;
and for any invalid request body (missing required fields, wrong types),
Pydantic validation must raise ValidationError.
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from pydantic import ValidationError

from app.models.requests import CompletionRequest, ChatMessage, ChatCompletionRequest


# ---------------------------------------------------------------------------
# Strategies for generating valid objects
# ---------------------------------------------------------------------------

valid_model_name = st.text(min_size=1, max_size=50).filter(lambda s: s.strip() != "")
valid_prompt = st.text(min_size=1, max_size=200).filter(lambda s: s.strip() != "")
valid_temperature = st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
valid_top_p = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
valid_max_tokens = st.integers(min_value=1, max_value=4096)
valid_role = st.sampled_from(["system", "user", "assistant"])
valid_content = st.text(min_size=1, max_size=200).filter(lambda s: s.strip() != "")

valid_chat_message = st.builds(
    ChatMessage,
    role=valid_role,
    content=valid_content,
)

valid_stop_list = st.one_of(
    st.none(),
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=3),
)

valid_completion_request = st.builds(
    CompletionRequest,
    model=valid_model_name,
    prompt=valid_prompt,
    max_tokens=st.one_of(st.none(), valid_max_tokens),
    temperature=st.one_of(st.none(), valid_temperature),
    top_p=st.one_of(st.none(), valid_top_p),
    stream=st.one_of(st.none(), st.booleans()),
    stop=valid_stop_list,
)

valid_chat_completion_request = st.builds(
    ChatCompletionRequest,
    model=valid_model_name,
    messages=st.lists(valid_chat_message, min_size=1, max_size=5),
    max_tokens=st.one_of(st.none(), valid_max_tokens),
    temperature=st.one_of(st.none(), valid_temperature),
    top_p=st.one_of(st.none(), valid_top_p),
    stream=st.one_of(st.none(), st.booleans()),
    stop=valid_stop_list,
)


# ---------------------------------------------------------------------------
# Property tests: valid objects pass validation
# ---------------------------------------------------------------------------


class TestValidObjectsPassValidation:
    """Valid request objects must always pass Pydantic validation."""

    @given(req=valid_completion_request)
    @settings(max_examples=100)
    def test_valid_completion_request_passes_validation(self, req: CompletionRequest):
        """**Validates: Requirements 3.5, 10.5**"""
        # Re-validate by constructing from dict — must not raise
        validated = CompletionRequest.model_validate(req.model_dump())
        assert validated.model == req.model
        assert validated.prompt == req.prompt

    @given(req=valid_chat_completion_request)
    @settings(max_examples=100)
    def test_valid_chat_completion_request_passes_validation(self, req: ChatCompletionRequest):
        """**Validates: Requirements 3.5, 10.5**"""
        validated = ChatCompletionRequest.model_validate(req.model_dump())
        assert validated.model == req.model
        assert len(validated.messages) == len(req.messages)

    @given(msg=valid_chat_message)
    @settings(max_examples=100)
    def test_valid_chat_message_passes_validation(self, msg: ChatMessage):
        """**Validates: Requirements 3.5, 10.5**"""
        validated = ChatMessage.model_validate(msg.model_dump())
        assert validated.role == msg.role
        assert validated.content == msg.content


# ---------------------------------------------------------------------------
# Property tests: invalid objects raise ValidationError
# ---------------------------------------------------------------------------


class TestInvalidObjectsRaiseValidationError:
    """Invalid request bodies must raise ValidationError."""

    @given(prompt=valid_prompt)
    @settings(max_examples=100)
    def test_completion_request_missing_model_raises(self, prompt: str):
        """**Validates: Requirements 3.6, 10.5**"""
        with pytest.raises(ValidationError):
            CompletionRequest.model_validate({"prompt": prompt})

    @given(model=valid_model_name)
    @settings(max_examples=100)
    def test_completion_request_missing_prompt_raises(self, model: str):
        """**Validates: Requirements 3.6, 10.5**"""
        with pytest.raises(ValidationError):
            CompletionRequest.model_validate({"model": model})

    @given(model=valid_model_name)
    @settings(max_examples=100)
    def test_chat_completion_request_missing_messages_raises(self, model: str):
        """**Validates: Requirements 3.6, 10.5**"""
        with pytest.raises(ValidationError):
            ChatCompletionRequest.model_validate({"model": model})

    @given(msg=valid_chat_message)
    @settings(max_examples=100)
    def test_chat_completion_request_missing_model_raises(self, msg: ChatMessage):
        """**Validates: Requirements 3.6, 10.5**"""
        with pytest.raises(ValidationError):
            ChatCompletionRequest.model_validate({"messages": [msg.model_dump()]})

    @given(
        model=valid_model_name,
        prompt=valid_prompt,
        bad_temp=st.one_of(
            st.lists(st.integers(), min_size=1, max_size=3),
            st.dictionaries(st.text(min_size=1, max_size=5), st.integers(), min_size=1, max_size=2),
        ),
    )
    @settings(max_examples=100)
    def test_completion_request_wrong_temperature_type_raises(self, model, prompt, bad_temp):
        """**Validates: Requirements 3.6, 10.5**"""
        with pytest.raises(ValidationError):
            CompletionRequest.model_validate(
                {"model": model, "prompt": prompt, "temperature": bad_temp}
            )

    @given(
        model=valid_model_name,
        prompt=valid_prompt,
        bad_max_tokens=st.one_of(
            st.lists(st.integers(), min_size=1, max_size=3),
            st.dictionaries(st.text(min_size=1, max_size=5), st.integers(), min_size=1, max_size=2),
        ),
    )
    @settings(max_examples=100)
    def test_completion_request_wrong_max_tokens_type_raises(self, model, prompt, bad_max_tokens):
        """**Validates: Requirements 3.6, 10.5**"""
        with pytest.raises(ValidationError):
            CompletionRequest.model_validate(
                {"model": model, "prompt": prompt, "max_tokens": bad_max_tokens}
            )

    @given(content=valid_content)
    @settings(max_examples=100)
    def test_chat_message_invalid_role_raises(self, content: str):
        """**Validates: Requirements 3.6, 10.5**"""
        with pytest.raises(ValidationError):
            ChatMessage.model_validate({"role": "invalid_role", "content": content})

    @given(role=valid_role)
    @settings(max_examples=100)
    def test_chat_message_missing_content_raises(self, role: str):
        """**Validates: Requirements 3.6, 10.5**"""
        with pytest.raises(ValidationError):
            ChatMessage.model_validate({"role": role})
