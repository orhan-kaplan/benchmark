# Feature: colab-benchmark-system, Property 1: Pydantic Model Doğrulama Tutarlılığı
# Validates: Requirements 1.1, 1.6, 2.2, 2.6
"""
Property-based tests for Pydantic model validation consistency.

For any valid field values, ModelEntry, Prompt, TestSet, BenchmarkRun,
ScoreEntry, and ResponseMetrics Pydantic models should be successfully
created and all required fields should be accessible; and for any data
dictionary with missing required fields, Pydantic should raise ValidationError.
"""

from datetime import datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from benchmark.models import (
    Backend,
    BenchmarkRun,
    BenchmarkRunConfig,
    Difficulty,
    GenerationParams,
    ModelEntry,
    Prompt,
    PromptCategory,
    PromptResult,
    ResponseMetrics,
    ScoreEntry,
    TestSet,
)

# --- Strategies ---

non_empty_text = st.text(min_size=1, max_size=50).filter(lambda s: s.strip())

backend_st = st.sampled_from(list(Backend))
category_st = st.sampled_from(list(PromptCategory))
difficulty_st = st.sampled_from(list(Difficulty))

model_entry_st = st.builds(
    ModelEntry,
    name=non_empty_text,
    repo=non_empty_text,
    format=non_empty_text,
    quantization=st.one_of(st.none(), non_empty_text),
    backend=backend_st,
    tags=st.lists(non_empty_text, max_size=5),
    api_endpoint=st.from_regex(r"https?://[a-z0-9]+\.[a-z]{2,4}", fullmatch=True),
)

prompt_st = st.builds(
    Prompt,
    id=non_empty_text,
    category=category_st,
    subcategory=st.one_of(st.none(), non_empty_text),
    text=non_empty_text,
    difficulty=difficulty_st,
    reference_answer=st.one_of(st.none(), non_empty_text),
)

test_set_st = st.builds(
    TestSet,
    name=non_empty_text,
    description=st.one_of(st.none(), non_empty_text),
    prompts=st.lists(prompt_st, max_size=5),
)

response_metrics_st = st.builds(
    ResponseMetrics,
    ttft_ms=st.one_of(st.none(), st.floats(min_value=0, max_value=1e6, allow_nan=False)),
    total_time_ms=st.floats(min_value=0.01, max_value=1e6, allow_nan=False),
    tokens_per_second=st.one_of(st.none(), st.floats(min_value=0, max_value=1e6, allow_nan=False)),
    prompt_tokens=st.integers(min_value=0, max_value=100000),
    completion_tokens=st.integers(min_value=0, max_value=100000),
)

score_entry_st = st.builds(
    ScoreEntry,
    prompt_id=non_empty_text,
    model_name=non_empty_text,
    manual_score=st.one_of(st.none(), st.integers(min_value=1, max_value=10)),
    judge_score=st.one_of(st.none(), st.floats(min_value=1, max_value=10, allow_nan=False)),
    comment=st.one_of(st.none(), non_empty_text),
)

benchmark_run_st = st.builds(
    BenchmarkRun,
    run_id=non_empty_text,
    timestamp=st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 1, 1),
    ),
    config=st.builds(
        BenchmarkRunConfig,
        test_set_name=non_empty_text,
        model_names=st.lists(non_empty_text, min_size=1, max_size=5),
        params=st.builds(GenerationParams),
    ),
    results=st.just([]),
    completed=st.booleans(),
)


# --- Property Tests: Valid data creates models and fields are accessible ---


@given(entry=model_entry_st)
@settings(max_examples=100)
def test_model_entry_valid_creation(entry: ModelEntry):
    """Valid data creates ModelEntry and all required fields are accessible."""
    assert isinstance(entry.name, str) and len(entry.name) > 0
    assert isinstance(entry.repo, str) and len(entry.repo) > 0
    assert isinstance(entry.format, str) and len(entry.format) > 0
    assert isinstance(entry.backend, Backend)
    assert isinstance(entry.tags, list)
    assert isinstance(entry.api_endpoint, str) and len(entry.api_endpoint) > 0


@given(prompt=prompt_st)
@settings(max_examples=100)
def test_prompt_valid_creation(prompt: Prompt):
    """Valid data creates Prompt and all required fields are accessible."""
    assert isinstance(prompt.id, str) and len(prompt.id) > 0
    assert isinstance(prompt.category, PromptCategory)
    assert isinstance(prompt.text, str) and len(prompt.text) > 0
    assert isinstance(prompt.difficulty, Difficulty)


@given(ts=test_set_st)
@settings(max_examples=100)
def test_test_set_valid_creation(ts: TestSet):
    """Valid data creates TestSet and all required fields are accessible."""
    assert isinstance(ts.name, str) and len(ts.name) > 0
    assert isinstance(ts.prompts, list)
    for p in ts.prompts:
        assert isinstance(p, Prompt)


@given(run=benchmark_run_st)
@settings(max_examples=100)
def test_benchmark_run_valid_creation(run: BenchmarkRun):
    """Valid data creates BenchmarkRun and all required fields are accessible."""
    assert isinstance(run.run_id, str) and len(run.run_id) > 0
    assert isinstance(run.timestamp, datetime)
    assert isinstance(run.config, BenchmarkRunConfig)
    assert isinstance(run.results, list)
    assert isinstance(run.completed, bool)


@given(se=score_entry_st)
@settings(max_examples=100)
def test_score_entry_valid_creation(se: ScoreEntry):
    """Valid data creates ScoreEntry and all required fields are accessible."""
    assert isinstance(se.prompt_id, str) and len(se.prompt_id) > 0
    assert isinstance(se.model_name, str) and len(se.model_name) > 0
    if se.manual_score is not None:
        assert 1 <= se.manual_score <= 10


@given(rm=response_metrics_st)
@settings(max_examples=100)
def test_response_metrics_valid_creation(rm: ResponseMetrics):
    """Valid data creates ResponseMetrics and all required fields are accessible."""
    assert isinstance(rm.total_time_ms, float)
    assert rm.total_time_ms >= 0
    assert isinstance(rm.prompt_tokens, int)
    assert isinstance(rm.completion_tokens, int)


# --- Property Tests: Missing required fields raise ValidationError ---


# Each model's required fields (no default value):
# ModelEntry: name, repo, format, backend, api_endpoint
# Prompt: id, category, text
# TestSet: name
# BenchmarkRun: run_id, timestamp, config
# ScoreEntry: prompt_id, model_name
# ResponseMetrics: total_time_ms

MODEL_ENTRY_REQUIRED = ["name", "repo", "format", "backend", "api_endpoint"]
PROMPT_REQUIRED = ["id", "category", "text"]
TEST_SET_REQUIRED = ["name"]
BENCHMARK_RUN_REQUIRED = ["run_id", "timestamp", "config"]
SCORE_ENTRY_REQUIRED = ["prompt_id", "model_name"]
RESPONSE_METRICS_REQUIRED = ["total_time_ms"]


@given(field=st.sampled_from(MODEL_ENTRY_REQUIRED))
@settings(max_examples=100)
def test_model_entry_missing_required_raises(field: str):
    """Missing a required field in ModelEntry raises ValidationError."""
    data = {
        "name": "test-model",
        "repo": "org/model",
        "format": "GGUF",
        "backend": "vllm",
        "api_endpoint": "http://localhost:8000",
    }
    del data[field]
    with pytest.raises(ValidationError):
        ModelEntry(**data)


@given(field=st.sampled_from(PROMPT_REQUIRED))
@settings(max_examples=100)
def test_prompt_missing_required_raises(field: str):
    """Missing a required field in Prompt raises ValidationError."""
    data = {
        "id": "p-001",
        "category": "kodlama",
        "text": "Write a function",
    }
    del data[field]
    with pytest.raises(ValidationError):
        Prompt(**data)


@given(field=st.sampled_from(TEST_SET_REQUIRED))
@settings(max_examples=100)
def test_test_set_missing_required_raises(field: str):
    """Missing a required field in TestSet raises ValidationError."""
    data = {"name": "my-set"}
    del data[field]
    with pytest.raises(ValidationError):
        TestSet(**data)


@given(field=st.sampled_from(BENCHMARK_RUN_REQUIRED))
@settings(max_examples=100)
def test_benchmark_run_missing_required_raises(field: str):
    """Missing a required field in BenchmarkRun raises ValidationError."""
    data = {
        "run_id": "run-001",
        "timestamp": datetime(2025, 1, 1),
        "config": {
            "test_set_name": "test",
            "model_names": ["m1"],
        },
    }
    del data[field]
    with pytest.raises(ValidationError):
        BenchmarkRun(**data)


@given(field=st.sampled_from(SCORE_ENTRY_REQUIRED))
@settings(max_examples=100)
def test_score_entry_missing_required_raises(field: str):
    """Missing a required field in ScoreEntry raises ValidationError."""
    data = {
        "prompt_id": "p-001",
        "model_name": "test-model",
    }
    del data[field]
    with pytest.raises(ValidationError):
        ScoreEntry(**data)


@given(field=st.sampled_from(RESPONSE_METRICS_REQUIRED))
@settings(max_examples=100)
def test_response_metrics_missing_required_raises(field: str):
    """Missing a required field in ResponseMetrics raises ValidationError."""
    data = {"total_time_ms": 100.0}
    del data[field]
    with pytest.raises(ValidationError):
        ResponseMetrics(**data)
