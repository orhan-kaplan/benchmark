"""Common test fixtures and Hypothesis profile configuration."""

from unittest.mock import AsyncMock

import pytest
from hypothesis import settings

from benchmark.api_client import APIClient
from benchmark.models import (
    APIResponse,
    Difficulty,
    ModelEntry,
    Prompt,
    PromptCategory,
    TestSet,
)

# --- Hypothesis profile ---

settings.register_profile("ci", max_examples=100)
settings.load_profile("ci")


# --- Fixtures ---


@pytest.fixture
def sample_model_entry() -> ModelEntry:
    """Return a sample ModelEntry for testing."""
    return ModelEntry(
        name="test-model",
        repo="org/test-model",
        format="FP16",
        quantization=None,
        backend="vllm",
        tags=["test", "çeviri"],
        api_endpoint="http://localhost:8000",
    )


@pytest.fixture
def sample_prompt() -> Prompt:
    """Return a sample Prompt for testing."""
    return Prompt(
        id="p-001",
        category=PromptCategory.TRANSLATION,
        text="Translate this sentence.",
        difficulty=Difficulty.EASY,
    )


@pytest.fixture
def sample_test_set(sample_prompt: Prompt) -> TestSet:
    """Return a sample TestSet with a few prompts."""
    return TestSet(
        name="test-set",
        description="Sample test set",
        prompts=[
            sample_prompt,
            Prompt(
                id="p-002",
                category=PromptCategory.CODING,
                text="Write a Python function.",
                difficulty=Difficulty.MEDIUM,
            ),
            Prompt(
                id="p-003",
                category=PromptCategory.REASONING,
                text="Solve this logic puzzle.",
                difficulty=Difficulty.HARD,
            ),
        ],
    )


@pytest.fixture
def mock_api_client() -> AsyncMock:
    """Return an AsyncMock of APIClient with a default successful response."""
    client = AsyncMock(spec=APIClient)
    client.chat_completion.return_value = APIResponse(
        status_code=200,
        response_text="Test response",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        error=None,
        elapsed_ms=500.0,
    )
    client.health_check.return_value = True
    return client
