# Feature: colab-benchmark-system, Property 11: Puan Aralığı Doğrulama
# Feature: colab-benchmark-system, Property 12: Puanlanmamış Öğe Filtreleme
# Validates: Requirements 5.1, 5.5, 5.6
"""
Property-based tests for ScoringEngine.

Property 11: For any integer, if the value is between 1 and 10 (inclusive)
the scoring engine should accept it; if outside this range it should raise ValueError.

Property 12: For any result set and for any scoring status distribution,
get_unscored should return only items where both manual_score and judge_score are None.
"""

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from benchmark.scoring import ScoringEngine
from benchmark.storage import StorageManager


# --- Strategies ---

valid_score = st.integers(min_value=1, max_value=10)
invalid_score_low = st.integers(max_value=0)
invalid_score_high = st.integers(min_value=11)
invalid_score = st.one_of(invalid_score_low, invalid_score_high)

prompt_id_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=1,
    max_size=20,
)
model_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=1,
    max_size=20,
)

# A result entry as stored in JSONL
result_entry_st = st.fixed_dictionaries({
    "prompt_id": prompt_id_st,
    "model_name": model_name_st,
    "response_text": st.text(min_size=0, max_size=50),
    "success": st.just(True),
})


# --- Fixtures ---

@pytest.fixture
def storage_and_engine(tmp_path_factory):
    """Create a fresh StorageManager + ScoringEngine pair."""
    base = tmp_path_factory.mktemp("scoring")
    storage = StorageManager(base)
    engine = ScoringEngine(storage=storage)
    return storage, engine


# --- Property 11: Puan Aralığı Doğrulama ---
# **Validates: Requirements 5.1, 5.6**


@given(score=valid_score)
@settings(max_examples=100)
def test_valid_scores_accepted(score: int, tmp_path_factory):
    """Scores in [1, 10] are accepted without error."""
    base = tmp_path_factory.mktemp("valid")
    storage = StorageManager(base)
    engine = ScoringEngine(storage=storage)

    run_id = "run_valid"
    storage.create_run_dir(run_id)
    storage.append_result(run_id, {
        "prompt_id": "p1",
        "model_name": "m1",
        "response_text": "test",
        "success": True,
    })

    engine.add_manual_score(run_id, "p1", "m1", score)
    scores = engine.get_scores(run_id)
    assert len(scores) == 1
    assert scores[0].manual_score == score


@given(score=invalid_score)
@settings(max_examples=100)
def test_invalid_scores_rejected(score: int, tmp_path_factory):
    """Scores outside [1, 10] raise ValueError."""
    base = tmp_path_factory.mktemp("invalid")
    storage = StorageManager(base)
    engine = ScoringEngine(storage=storage)

    run_id = "run_invalid"
    storage.create_run_dir(run_id)

    with pytest.raises(ValueError):
        engine.add_manual_score(run_id, "p1", "m1", score)


# --- Property 12: Puanlanmamış Öğe Filtreleme ---
# **Validates: Requirements 5.5**


# Scoring action: None (unscored), "manual", "judge", or "both"
scoring_action_st = st.sampled_from([None, "manual", "judge", "both"])


@given(
    num_results=st.integers(min_value=1, max_value=10),
    scoring_actions=st.lists(scoring_action_st, min_size=1, max_size=10),
)
@settings(max_examples=100)
def test_get_unscored_returns_only_unscored_items(
    num_results: int,
    scoring_actions: list,
    tmp_path_factory,
):
    """get_unscored returns exactly the items where both manual_score and judge_score are None."""
    base = tmp_path_factory.mktemp("unscored")
    storage = StorageManager(base)
    engine = ScoringEngine(storage=storage)

    run_id = "run_unscored"
    storage.create_run_dir(run_id)

    # Align scoring_actions length with num_results
    actions = scoring_actions[:num_results]
    while len(actions) < num_results:
        actions.append(None)

    # Create distinct results
    results = []
    for i in range(num_results):
        result = {
            "prompt_id": f"p{i}",
            "model_name": f"m{i}",
            "response_text": f"response {i}",
            "success": True,
        }
        storage.append_result(run_id, result)
        results.append(result)

    # Apply scoring actions by writing scores.json directly
    # (manual scores via engine, judge scores via direct file write)
    scores_list = []
    expected_unscored_keys = set()

    for i, action in enumerate(actions):
        pid = f"p{i}"
        mname = f"m{i}"
        key = (pid, mname)

        if action == "manual":
            scores_list.append({
                "prompt_id": pid,
                "model_name": mname,
                "manual_score": 5,
                "judge_score": None,
                "comment": None,
            })
        elif action == "judge":
            scores_list.append({
                "prompt_id": pid,
                "model_name": mname,
                "manual_score": None,
                "judge_score": 7.0,
                "comment": None,
            })
        elif action == "both":
            scores_list.append({
                "prompt_id": pid,
                "model_name": mname,
                "manual_score": 8,
                "judge_score": 6.5,
                "comment": None,
            })
        else:
            # None — no score entry, should be unscored
            expected_unscored_keys.add(key)

    # Write scores directly to scores.json
    scores_path = storage.runs_dir / run_id / "scores.json"
    storage.write_json(scores_path, {"scores": scores_list})

    # Get unscored items
    unscored = engine.get_unscored(run_id)
    unscored_keys = {(u["prompt_id"], u["model_name"]) for u in unscored}

    assert unscored_keys == expected_unscored_keys
