# Feature: colab-benchmark-system, Property 2: Veri Serileştirme Round-Trip
# Validates: Requirements 1.3, 2.3, 8.1, 8.3, 8.4, 8.5, 8.6, 11.1, 11.2, 11.3, 12.1, 12.3
"""
Property-based tests for data serialization round-trip.

For any valid ModelEntry, TestSet, BenchmarkRun, ScoreEntry, GenerationParams,
or ResponseMetrics object, serializing to JSON (model_dump_json) and parsing
back (model_validate_json) should produce an equivalent object.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from benchmark.models import (
    BenchmarkRun,
    GenerationParams,
    ModelEntry,
    ResponseMetrics,
    ScoreEntry,
    TestSet,
)
from tests.test_property_models import (
    benchmark_run_st,
    model_entry_st,
    response_metrics_st,
    score_entry_st,
    test_set_st,
)

# --- Strategy for GenerationParams (not in test_property_models.py) ---

generation_params_st = st.builds(
    GenerationParams,
    temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
    max_tokens=st.integers(min_value=1, max_value=100000),
    top_p=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    repetition_count=st.integers(min_value=1, max_value=10),
)


# --- Round-Trip Property Tests ---


@given(entry=model_entry_st)
@settings(max_examples=100)
def test_model_entry_roundtrip(entry: ModelEntry):
    """Serializing and deserializing ModelEntry produces an equivalent object."""
    json_str = entry.model_dump_json()
    restored = ModelEntry.model_validate_json(json_str)
    assert restored == entry


@given(ts=test_set_st)
@settings(max_examples=100)
def test_test_set_roundtrip(ts: TestSet):
    """Serializing and deserializing TestSet produces an equivalent object."""
    json_str = ts.model_dump_json()
    restored = TestSet.model_validate_json(json_str)
    assert restored == ts


@given(run=benchmark_run_st)
@settings(max_examples=100)
def test_benchmark_run_roundtrip(run: BenchmarkRun):
    """Serializing and deserializing BenchmarkRun produces an equivalent object."""
    json_str = run.model_dump_json()
    restored = BenchmarkRun.model_validate_json(json_str)
    assert restored == run


@given(se=score_entry_st)
@settings(max_examples=100)
def test_score_entry_roundtrip(se: ScoreEntry):
    """Serializing and deserializing ScoreEntry produces an equivalent object."""
    json_str = se.model_dump_json()
    restored = ScoreEntry.model_validate_json(json_str)
    assert restored == se


@given(params=generation_params_st)
@settings(max_examples=100)
def test_generation_params_roundtrip(params: GenerationParams):
    """Serializing and deserializing GenerationParams produces an equivalent object."""
    json_str = params.model_dump_json()
    restored = GenerationParams.model_validate_json(json_str)
    assert restored == params


@given(rm=response_metrics_st)
@settings(max_examples=100)
def test_response_metrics_roundtrip(rm: ResponseMetrics):
    """Serializing and deserializing ResponseMetrics produces an equivalent object."""
    json_str = rm.model_dump_json()
    restored = ResponseMetrics.model_validate_json(json_str)
    assert restored == rm
