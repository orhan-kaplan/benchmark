# Feature: colab-benchmark-system, Property 3: Model Kataloğu Benzersizlik Kısıtı
# Feature: colab-benchmark-system, Property 4: Model Kataloğu Filtreleme Doğruluğu
# Validates: Requirements 1.2, 1.4, 1.5
"""
Property-based tests for ModelCatalog uniqueness constraint and filtering correctness.

Property 3: For any list of models, if two models have the same name, the second
registration should be rejected (ValueError); all models with different names should
be registered successfully.

Property 4: For any set of registered models and for any backend/tag filter
combination, all returned models should satisfy the filter criteria and no model
satisfying the filter criteria should be excluded from the results.
"""

from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from benchmark.catalog import ModelCatalog
from benchmark.models import Backend, ModelEntry

# --- Strategies ---

backend_st = st.sampled_from(list(Backend))
tag_st = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz"),
    min_size=1,
    max_size=8,
)
tags_st = st.lists(tag_st, max_size=4)
model_name_st = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789-"),
    min_size=1,
    max_size=20,
).filter(lambda s: s.strip() and not s.startswith("-"))


def make_entry(name: str, backend: Backend = Backend.VLLM, tags: list[str] | None = None) -> ModelEntry:
    return ModelEntry(
        name=name,
        repo="org/model",
        format="FP16",
        backend=backend,
        tags=tags or [],
        api_endpoint="http://localhost:8000",
    )


# --- Property 3: Model Kataloğu Benzersizlik Kısıtı ---
# **Validates: Requirements 1.2**


@given(
    names=st.lists(model_name_st, min_size=1, max_size=10),
)
@settings(max_examples=100)
def test_uniqueness_constraint(tmp_path_factory, names: list[str]):
    """For any list of models, duplicate names are rejected and unique names succeed.

    **Validates: Requirements 1.2**
    """
    tmp_path = tmp_path_factory.mktemp("catalog")
    catalog = ModelCatalog(tmp_path)

    seen: set[str] = set()
    for name in names:
        entry = make_entry(name)
        if name in seen:
            # Duplicate — must raise ValueError
            with pytest.raises(ValueError):
                catalog.register(entry)
        else:
            # First occurrence — must succeed
            catalog.register(entry)
            seen.add(name)

    # All unique names should be registered
    registered = {m.name for m in catalog.list_all()}
    assert registered == seen


# --- Property 4: Model Kataloğu Filtreleme Doğruluğu ---
# **Validates: Requirements 1.4, 1.5**


@given(
    models_data=st.lists(
        st.tuples(model_name_st, backend_st, tags_st),
        min_size=1,
        max_size=8,
    ),
    filter_backend=st.one_of(st.none(), backend_st),
    filter_tags=st.one_of(st.none(), st.lists(tag_st, min_size=1, max_size=3)),
)
@settings(max_examples=100)
def test_filter_correctness(
    tmp_path_factory,
    models_data: list[tuple[str, Backend, list[str]]],
    filter_backend: Backend | None,
    filter_tags: list[str] | None,
):
    """All returned models satisfy the filter and no qualifying model is excluded.

    **Validates: Requirements 1.4, 1.5**
    """
    # Deduplicate names so we can register all of them
    seen_names: set[str] = set()
    unique_models: list[tuple[str, Backend, list[str]]] = []
    for name, backend, tags in models_data:
        if name not in seen_names:
            seen_names.add(name)
            unique_models.append((name, backend, tags))

    assume(len(unique_models) >= 1)

    tmp_path = tmp_path_factory.mktemp("catalog")
    catalog = ModelCatalog(tmp_path)

    for name, backend, tags in unique_models:
        catalog.register(make_entry(name, backend=backend, tags=tags))

    # Apply filter
    backend_val = filter_backend.value if filter_backend is not None else None
    result = catalog.filter_by(backend=backend_val, tags=filter_tags)
    result_names = {m.name for m in result}

    # Compute expected set manually
    expected_names: set[str] = set()
    for name, backend, tags in unique_models:
        matches_backend = filter_backend is None or backend == filter_backend
        matches_tags = filter_tags is None or all(t in tags for t in filter_tags)
        if matches_backend and matches_tags:
            expected_names.add(name)

    # All returned models satisfy the filter (no false positives)
    for m in result:
        if filter_backend is not None:
            assert m.backend == filter_backend
        if filter_tags is not None:
            for t in filter_tags:
                assert t in m.tags

    # No qualifying model is excluded (no false negatives)
    assert result_names == expected_names
