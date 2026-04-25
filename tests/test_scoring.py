"""Birim testleri — ScoringEngine."""

import pytest

from benchmark.models import ScoreEntry
from benchmark.scoring import ScoringEngine
from benchmark.storage import StorageManager


@pytest.fixture
def storage(tmp_path):
    return StorageManager(tmp_path)


@pytest.fixture
def engine(storage):
    return ScoringEngine(storage=storage)


@pytest.fixture
def run_with_results(storage):
    """Create a run with some results for testing."""
    run_id = "test_run_001"
    storage.create_run_dir(run_id)
    storage.append_result(run_id, {
        "prompt_id": "p1",
        "model_name": "modelA",
        "response_text": "Hello world",
        "success": True,
    })
    storage.append_result(run_id, {
        "prompt_id": "p2",
        "model_name": "modelA",
        "response_text": "Goodbye",
        "success": True,
    })
    storage.append_result(run_id, {
        "prompt_id": "p1",
        "model_name": "modelB",
        "response_text": "Hi there",
        "success": True,
    })
    return run_id


class TestAddManualScore:
    def test_valid_score(self, engine, storage, run_with_results):
        run_id = run_with_results
        engine.add_manual_score(run_id, "p1", "modelA", 7, "Good answer")
        scores = engine.get_scores(run_id)
        assert len(scores) == 1
        assert scores[0].manual_score == 7
        assert scores[0].comment == "Good answer"

    def test_score_boundaries(self, engine, run_with_results):
        run_id = run_with_results
        engine.add_manual_score(run_id, "p1", "modelA", 1)
        engine.add_manual_score(run_id, "p2", "modelA", 10)
        scores = engine.get_scores(run_id)
        assert scores[0].manual_score == 1
        assert scores[1].manual_score == 10

    def test_score_below_range(self, engine, run_with_results):
        with pytest.raises(ValueError, match="between 1 and 10"):
            engine.add_manual_score(run_with_results, "p1", "modelA", 0)

    def test_score_above_range(self, engine, run_with_results):
        with pytest.raises(ValueError, match="between 1 and 10"):
            engine.add_manual_score(run_with_results, "p1", "modelA", 11)

    def test_update_existing_score(self, engine, run_with_results):
        run_id = run_with_results
        engine.add_manual_score(run_id, "p1", "modelA", 5, "OK")
        engine.add_manual_score(run_id, "p1", "modelA", 8, "Better")
        scores = engine.get_scores(run_id)
        assert len(scores) == 1
        assert scores[0].manual_score == 8
        assert scores[0].comment == "Better"

    def test_score_without_comment(self, engine, run_with_results):
        run_id = run_with_results
        engine.add_manual_score(run_id, "p1", "modelA", 6)
        scores = engine.get_scores(run_id)
        assert scores[0].comment is None


class TestGetUnscored:
    def test_all_unscored_initially(self, engine, run_with_results):
        unscored = engine.get_unscored(run_with_results)
        assert len(unscored) == 3

    def test_scored_items_excluded(self, engine, run_with_results):
        run_id = run_with_results
        engine.add_manual_score(run_id, "p1", "modelA", 7)
        unscored = engine.get_unscored(run_id)
        assert len(unscored) == 2
        prompt_model_pairs = [(u["prompt_id"], u["model_name"]) for u in unscored]
        assert ("p1", "modelA") not in prompt_model_pairs

    def test_empty_run(self, engine, storage):
        run_id = "empty_run"
        storage.create_run_dir(run_id)
        unscored = engine.get_unscored(run_id)
        assert unscored == []


class TestGetScores:
    def test_empty_scores(self, engine, storage):
        run_id = "no_scores"
        storage.create_run_dir(run_id)
        scores = engine.get_scores(run_id)
        assert scores == []

    def test_returns_score_entries(self, engine, run_with_results):
        run_id = run_with_results
        engine.add_manual_score(run_id, "p1", "modelA", 5)
        scores = engine.get_scores(run_id)
        assert all(isinstance(s, ScoreEntry) for s in scores)


class TestParseScore:
    def test_integer(self):
        assert ScoringEngine._parse_score("7") == 7.0

    def test_float(self):
        assert ScoringEngine._parse_score("8.5") == 8.5

    def test_with_text(self):
        assert ScoringEngine._parse_score("The score is 6 out of 10") == 6.0

    def test_clamp_high(self):
        assert ScoringEngine._parse_score("15") == 10.0

    def test_clamp_low(self):
        assert ScoringEngine._parse_score("0") == 1.0

    def test_none_input(self):
        assert ScoringEngine._parse_score(None) is None

    def test_empty_string(self):
        assert ScoringEngine._parse_score("") is None

    def test_no_number(self):
        assert ScoringEngine._parse_score("no score here") is None
