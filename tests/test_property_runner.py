# Feature: colab-benchmark-system, Property 7: Benchmark Runner Tamlık ve Adillik
# Feature: colab-benchmark-system, Property 8: Benchmark Runner Hata Dayanıklılığı
# Feature: colab-benchmark-system, Property 19: Parametre Geçersiz Kılma
# Validates: Requirements 3.1, 3.2, 3.4, 3.7, 10.1, 10.3, 10.4
"""
Property-based tests for BenchmarkRunner.

Property 7: For any test set (N prompts) and model list (M models), when the
benchmark run completes, the result count should be N × M, and for each model
the same prompts should be sent with identical parameters.

Property 8: For any prompt-model pair sequence and for any subset of failed API
responses, the runner should log failures and continue producing valid results
for successful ones; total result count should still be N × M.

Property 19: For any default GenerationParams and for any override parameters,
the effective parameters should reflect the override values; and the used
parameters should be recorded in the run metadata.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from benchmark.api_client import APIClient
from benchmark.catalog import ModelCatalog
from benchmark.metrics import MetricsCollector
from benchmark.models import (
    APIResponse,
    Backend,
    BenchmarkRunConfig,
    Difficulty,
    GenerationParams,
    ModelEntry,
    Prompt,
    PromptCategory,
    TestSet,
)
from benchmark.runner import BenchmarkRunner
from benchmark.storage import StorageManager
from benchmark.test_sets import TestSetManager


# --- Strategies ---

# Strategy for generating unique prompt IDs
prompt_id_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=1,
    max_size=20,
)

# Strategy for generating model names (ASCII-safe)
model_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=1,
    max_size=20,
)

category_st = st.sampled_from(list(PromptCategory))
difficulty_st = st.sampled_from(list(Difficulty))

prompt_st = st.builds(
    Prompt,
    id=prompt_id_st,
    category=category_st,
    text=st.text(min_size=1, max_size=50),
    difficulty=difficulty_st,
)

# Strategy for GenerationParams with reasonable ranges
generation_params_st = st.builds(
    GenerationParams,
    temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    max_tokens=st.integers(min_value=1, max_value=4096),
    top_p=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    repetition_count=st.just(1),  # Keep repetition_count=1 for simplicity
)


def make_mock_api_client(fail_indices: set[int] | None = None) -> AsyncMock:
    """Create a mock APIClient that optionally fails at specific call indices.

    Args:
        fail_indices: Set of 0-based call indices that should return errors.
                      If None, all calls succeed.
    """
    client = AsyncMock(spec=APIClient)
    call_counter = {"n": 0}

    async def mock_chat_completion(endpoint, model, messages, params):
        idx = call_counter["n"]
        call_counter["n"] += 1

        if fail_indices and idx in fail_indices:
            return APIResponse(
                status_code=500,
                response_text=None,
                usage=None,
                error=f"Simulated failure at call {idx}",
                elapsed_ms=50.0,
            )
        return APIResponse(
            status_code=200,
            response_text="Generated response text",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            error=None,
            elapsed_ms=100.0,
        )

    client.chat_completion.side_effect = mock_chat_completion
    return client


def setup_runner(
    tmp_path: Path,
    prompts: list[Prompt],
    model_names: list[str],
    fail_indices: set[int] | None = None,
) -> tuple[BenchmarkRunner, StorageManager]:
    """Set up a BenchmarkRunner with given prompts and models.

    Returns the runner and storage manager for assertions.
    """
    data_path = tmp_path / "benchmark_data"
    data_path.mkdir(parents=True, exist_ok=True)

    storage = StorageManager(data_path)
    catalog = ModelCatalog(data_path)
    test_set_mgr = TestSetManager(data_path)

    # Register models
    for name in model_names:
        catalog.register(
            ModelEntry(
                name=name,
                repo=f"org/{name}",
                format="FP16",
                backend=Backend.VLLM,
                tags=["test"],
                api_endpoint=f"http://localhost:8000/{name}",
            )
        )

    # Create test set
    test_set = TestSet(name="prop-test-set", description="Property test set", prompts=prompts)
    test_set_mgr.create(test_set)

    mock_client = make_mock_api_client(fail_indices)
    metrics = MetricsCollector()

    runner = BenchmarkRunner(
        catalog=catalog,
        test_set_manager=test_set_mgr,
        api_client=mock_client,
        metrics_collector=metrics,
        storage=storage,
    )
    return runner, storage


# ---------------------------------------------------------------------------
# Property 7: Benchmark Runner Tamlık ve Adillik
# ---------------------------------------------------------------------------


@given(
    num_prompts=st.integers(min_value=1, max_value=5),
    num_models=st.integers(min_value=1, max_value=4),
    params=generation_params_st,
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_runner_completeness_result_count(
    tmp_path_factory, num_prompts: int, num_models: int, params: GenerationParams
):
    """**Validates: Requirements 3.1, 3.2, 3.7**

    Property 7 — Result count: For any N prompts and M models, the benchmark
    run should produce exactly N × M results.
    """
    tmp_path = tmp_path_factory.mktemp("prop7_count")

    # Generate unique prompts
    prompts = [
        Prompt(id=f"p{i}", category=PromptCategory.CODING, text=f"Prompt {i}")
        for i in range(num_prompts)
    ]
    model_names = [f"model-{i}" for i in range(num_models)]

    runner, storage = setup_runner(tmp_path, prompts, model_names)

    config = BenchmarkRunConfig(
        test_set_name="prop-test-set",
        model_names=model_names,
        params=params,
    )
    result = await runner.run(config)

    expected_count = num_prompts * num_models
    assert len(result.results) == expected_count, (
        f"Expected {expected_count} results (N={num_prompts} × M={num_models}), "
        f"got {len(result.results)}"
    )
    assert result.completed is True


@given(
    num_prompts=st.integers(min_value=1, max_value=5),
    num_models=st.integers(min_value=1, max_value=4),
    params=generation_params_st,
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_runner_fairness_identical_params(
    tmp_path_factory, num_prompts: int, num_models: int, params: GenerationParams
):
    """**Validates: Requirements 3.1, 3.2, 3.7**

    Property 7 — Fairness: For each model, the same prompts should be sent
    with identical parameters (temperature, max_tokens, top_p).
    """
    tmp_path = tmp_path_factory.mktemp("prop7_fair")

    prompts = [
        Prompt(id=f"p{i}", category=PromptCategory.CODING, text=f"Prompt {i}")
        for i in range(num_prompts)
    ]
    model_names = [f"model-{i}" for i in range(num_models)]

    runner, storage = setup_runner(tmp_path, prompts, model_names)

    config = BenchmarkRunConfig(
        test_set_name="prop-test-set",
        model_names=model_names,
        params=params,
    )
    result = await runner.run(config)

    # Verify each model received the same set of prompt IDs
    results_by_model: dict[str, list[str]] = {}
    for r in result.results:
        results_by_model.setdefault(r.model_name, []).append(r.prompt_id)

    prompt_ids = [p.id for p in prompts]
    for model_name, received_ids in results_by_model.items():
        assert sorted(received_ids) == sorted(prompt_ids), (
            f"Model {model_name} did not receive all prompts. "
            f"Expected {sorted(prompt_ids)}, got {sorted(received_ids)}"
        )

    # Verify API was called with identical params for each prompt across models
    calls = runner.api_client.chat_completion.call_args_list
    for call in calls:
        _, kwargs = call
        call_params = kwargs.get("params") or call[0][3] if len(call[0]) > 3 else kwargs["params"]
        assert call_params.temperature == params.temperature
        assert call_params.max_tokens == params.max_tokens
        assert call_params.top_p == params.top_p


# ---------------------------------------------------------------------------
# Property 8: Benchmark Runner Hata Dayanıklılığı
# ---------------------------------------------------------------------------


@given(
    num_prompts=st.integers(min_value=1, max_value=5),
    num_models=st.integers(min_value=1, max_value=4),
    data=st.data(),
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_runner_error_resilience(
    tmp_path_factory, num_prompts: int, num_models: int, data
):
    """**Validates: Requirements 3.4**

    Property 8: For any prompt-model pair sequence and for any subset of failed
    API responses, the runner should log failures and continue producing valid
    results for successful ones; total result count should still be N × M.
    """
    tmp_path = tmp_path_factory.mktemp("prop8")

    prompts = [
        Prompt(id=f"p{i}", category=PromptCategory.CODING, text=f"Prompt {i}")
        for i in range(num_prompts)
    ]
    model_names = [f"model-{i}" for i in range(num_models)]

    total_calls = num_prompts * num_models

    # Draw a random subset of call indices to fail
    fail_indices = data.draw(
        st.frozensets(st.integers(min_value=0, max_value=total_calls - 1)),
        label="fail_indices",
    )

    runner, storage = setup_runner(tmp_path, prompts, model_names, fail_indices=set(fail_indices))

    config = BenchmarkRunConfig(
        test_set_name="prop-test-set",
        model_names=model_names,
    )
    result = await runner.run(config)

    # Total result count must always be N × M regardless of failures
    assert len(result.results) == total_calls, (
        f"Expected {total_calls} results, got {len(result.results)}"
    )
    assert result.completed is True

    # Failed results should have success=False and error set
    failed_results = [r for r in result.results if not r.success]
    successful_results = [r for r in result.results if r.success]

    assert len(failed_results) == len(fail_indices), (
        f"Expected {len(fail_indices)} failures, got {len(failed_results)}"
    )
    assert len(successful_results) == total_calls - len(fail_indices)

    # Every failed result must have an error message
    for r in failed_results:
        assert r.error is not None, "Failed result must have an error message"

    # Every successful result must have response_text
    for r in successful_results:
        assert r.response_text is not None, "Successful result must have response_text"


# ---------------------------------------------------------------------------
# Property 19: Parametre Geçersiz Kılma (Override)
# ---------------------------------------------------------------------------


@given(
    default_params=generation_params_st,
    override_temp=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    override_max_tokens=st.integers(min_value=1, max_value=4096),
    override_top_p=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_parameter_override_effective(
    tmp_path_factory,
    default_params: GenerationParams,
    override_temp: float,
    override_max_tokens: int,
    override_top_p: float,
):
    """**Validates: Requirements 10.1, 10.3, 10.4**

    Property 19 — Override effectiveness: For any default GenerationParams and
    for any override parameters, the effective parameters should reflect the
    override values.
    """
    tmp_path = tmp_path_factory.mktemp("prop19_eff")

    prompts = [Prompt(id="p0", category=PromptCategory.GENERAL, text="Test")]
    model_names = ["model-0"]

    runner, storage = setup_runner(tmp_path, prompts, model_names)

    # Create config with override params (simulating CLI override of defaults)
    override_params = GenerationParams(
        temperature=override_temp,
        max_tokens=override_max_tokens,
        top_p=override_top_p,
        repetition_count=1,
    )
    config = BenchmarkRunConfig(
        test_set_name="prop-test-set",
        model_names=model_names,
        params=override_params,
    )
    result = await runner.run(config)

    # Verify the API was called with the override params, not defaults
    call_args = runner.api_client.chat_completion.call_args_list
    assert len(call_args) == 1

    _, kwargs = call_args[0]
    used_params = kwargs.get("params") or call_args[0][0][3]
    assert used_params.temperature == override_temp
    assert used_params.max_tokens == override_max_tokens
    assert used_params.top_p == override_top_p

    # Verify the config in the result reflects override values
    assert result.config.params.temperature == override_temp
    assert result.config.params.max_tokens == override_max_tokens
    assert result.config.params.top_p == override_top_p


@given(
    override_params=generation_params_st,
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_parameter_override_recorded_in_metadata(
    tmp_path_factory,
    override_params: GenerationParams,
):
    """**Validates: Requirements 10.1, 10.3, 10.4**

    Property 19 — Metadata recording: The used parameters should be recorded
    in the run metadata (meta.json).
    """
    tmp_path = tmp_path_factory.mktemp("prop19_meta")

    prompts = [Prompt(id="p0", category=PromptCategory.GENERAL, text="Test")]
    model_names = ["model-0"]

    runner, storage = setup_runner(tmp_path, prompts, model_names)

    config = BenchmarkRunConfig(
        test_set_name="prop-test-set",
        model_names=model_names,
        params=override_params,
    )
    result = await runner.run(config)

    # Read meta.json and verify params are recorded
    meta = storage.read_json(storage.runs_dir / result.run_id / "meta.json")

    assert meta["config"]["params"]["temperature"] == override_params.temperature
    assert meta["config"]["params"]["max_tokens"] == override_params.max_tokens
    assert meta["config"]["params"]["top_p"] == override_params.top_p
    assert meta["config"]["params"]["repetition_count"] == override_params.repetition_count
