# Feature: colab-benchmark-system, Property 5: Test Seti Prompt Ekleme İnvariantı
# Feature: colab-benchmark-system, Property 6: Test Seti Filtreleme Doğruluğu
# Validates: Requirements 2.4, 2.5
"""
Property-based tests for TestSetManager prompt addition and filtering.

Property 5: For any test set and for any valid prompt, after adding the prompt,
the test set's prompt count should increase by exactly one and the added prompt
should be findable in the test set.

Property 6: For any set of prompts and for any category/difficulty filter
combination, all returned prompts should satisfy the filter criteria and no
prompt satisfying the filter criteria should be excluded from the results.
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from benchmark.models import Difficulty, Prompt, PromptCategory, TestSet
from benchmark.test_sets import TestSetManager

# --- Strategies ---

category_st = st.sampled_from(list(PromptCategory))
difficulty_st = st.sampled_from(list(Difficulty))
non_empty_text = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789 "),
    min_size=1,
    max_size=30,
).filter(lambda s: s.strip())

prompt_id_st = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789-"),
    min_size=1,
    max_size=15,
).filter(lambda s: s.strip() and not s.startswith("-"))

prompt_st = st.builds(
    Prompt,
    id=prompt_id_st,
    category=category_st,
    text=non_empty_text,
    difficulty=difficulty_st,
)


# --- Property 5: Test Seti Prompt Ekleme İnvariantı ---
# **Validates: Requirements 2.4**


@given(
    existing_prompts=st.lists(prompt_st, max_size=5),
    new_prompt=prompt_st,
)
@settings(max_examples=100)
def test_add_prompt_invariant(tmp_path_factory, existing_prompts, new_prompt):
    """After adding a prompt, count increases by one and the prompt is findable.

    **Validates: Requirements 2.4**
    """
    # Ensure all existing prompt IDs are unique
    seen_ids: set[str] = set()
    unique_prompts: list[Prompt] = []
    for p in existing_prompts:
        if p.id not in seen_ids:
            seen_ids.add(p.id)
            unique_prompts.append(p)

    # The new prompt must have a unique ID (not in existing set)
    assume(new_prompt.id not in seen_ids)

    tmp_path = tmp_path_factory.mktemp("testsets")
    manager = TestSetManager(data_path=tmp_path)

    ts = TestSet(name="test", prompts=unique_prompts)
    manager.create(ts)

    count_before = len(manager.load("test").prompts)

    manager.add_prompt("test", new_prompt)

    loaded = manager.load("test")
    count_after = len(loaded.prompts)

    # Count increased by exactly one
    assert count_after == count_before + 1

    # The added prompt is findable
    found = [p for p in loaded.prompts if p.id == new_prompt.id]
    assert len(found) == 1
    assert found[0].text == new_prompt.text
    assert found[0].category == new_prompt.category
    assert found[0].difficulty == new_prompt.difficulty


# --- Property 6: Test Seti Filtreleme Doğruluğu ---
# **Validates: Requirements 2.5**


@given(
    prompts=st.lists(prompt_st, min_size=1, max_size=10),
    filter_category=st.one_of(st.none(), category_st),
    filter_difficulty=st.one_of(st.none(), difficulty_st),
)
@settings(max_examples=100)
def test_filter_prompts_correctness(
    tmp_path_factory, prompts, filter_category, filter_difficulty
):
    """All returned prompts satisfy the filter and no qualifying prompt is excluded.

    **Validates: Requirements 2.5**
    """
    # Deduplicate by ID
    seen_ids: set[str] = set()
    unique_prompts: list[Prompt] = []
    for p in prompts:
        if p.id not in seen_ids:
            seen_ids.add(p.id)
            unique_prompts.append(p)

    assume(len(unique_prompts) >= 1)

    tmp_path = tmp_path_factory.mktemp("testsets")
    manager = TestSetManager(data_path=tmp_path)

    ts = TestSet(name="filter_test", prompts=unique_prompts)
    manager.create(ts)

    cat_val = filter_category.value if filter_category is not None else None
    diff_val = filter_difficulty.value if filter_difficulty is not None else None
    result = manager.filter_prompts("filter_test", category=cat_val, difficulty=diff_val)

    result_ids = {p.id for p in result}

    # Compute expected set manually
    expected_ids: set[str] = set()
    for p in unique_prompts:
        matches_cat = filter_category is None or p.category == filter_category
        matches_diff = filter_difficulty is None or p.difficulty == filter_difficulty
        if matches_cat and matches_diff:
            expected_ids.add(p.id)

    # All returned prompts satisfy the filter (no false positives)
    for p in result:
        if filter_category is not None:
            assert p.category == filter_category
        if filter_difficulty is not None:
            assert p.difficulty == filter_difficulty

    # No qualifying prompt is excluded (no false negatives)
    assert result_ids == expected_ids
