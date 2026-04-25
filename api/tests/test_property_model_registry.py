# Feature: vllm-docker-inference, Property 7: Model Registry Tutarlılığı
"""
Property-based tests for Model Registry consistency.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4**

Property 7: For any set of model name and endpoint pairs, after all pairs
are registered: (a) looking up a registered model name returns the correct
endpoint, (b) looking up an unregistered model name raises an error,
(c) the model list contains all registered models.
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.services.model_registry import ModelNotFoundError, ModelRegistry
from app.models.responses import ModelInfo


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

model_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() != "")

endpoint_st = st.from_regex(
    r"http://[a-z][a-z0-9\-]{0,19}:[0-9]{4,5}", fullmatch=True
)

model_entry_st = st.tuples(model_name_st, endpoint_st)

# List of unique model entries (unique by name)
model_entries_st = st.lists(
    model_entry_st,
    min_size=1,
    max_size=20,
    unique_by=lambda entry: entry[0],
)

unregistered_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() != "")


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestModelRegistryConsistency:
    """Property 7: Model Registry Tutarlılığı."""

    @given(entries=model_entries_st)
    @settings(max_examples=100)
    def test_registered_model_returns_correct_endpoint(self, entries):
        """After registering N models, each returns its correct endpoint.

        **Validates: Requirements 7.1, 7.2**
        """
        registry = ModelRegistry()
        for name, endpoint in entries:
            registry.register_model(name, endpoint)

        for name, endpoint in entries:
            assert registry.get_endpoint(name) == endpoint

    @given(entries=model_entries_st, unknown=unregistered_name_st)
    @settings(max_examples=100)
    def test_unregistered_model_raises_error(self, entries, unknown):
        """Looking up an unregistered model raises ModelNotFoundError.

        **Validates: Requirements 7.3**
        """
        registered_names = {name for name, _ in entries}
        assume(unknown not in registered_names)

        registry = ModelRegistry()
        for name, endpoint in entries:
            registry.register_model(name, endpoint)

        with pytest.raises(ModelNotFoundError) as exc_info:
            registry.get_endpoint(unknown)

        assert exc_info.value.model_name == unknown
        assert set(exc_info.value.available_models) == registered_names

    @given(entries=model_entries_st)
    @settings(max_examples=100)
    def test_list_models_contains_all_registered(self, entries):
        """list_models() returns all registered models.

        **Validates: Requirements 7.4**
        """
        registry = ModelRegistry()
        for name, endpoint in entries:
            registry.register_model(name, endpoint)

        models = registry.list_models()
        assert len(models) == len(entries)
        assert all(isinstance(m, ModelInfo) for m in models)

        returned_ids = {m.id for m in models}
        expected_ids = {name for name, _ in entries}
        assert returned_ids == expected_ids
