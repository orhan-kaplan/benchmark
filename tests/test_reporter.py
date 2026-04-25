"""Birim testleri — ReportGenerator."""

import json
from pathlib import Path

import pytest

from benchmark.reporter import ReportGenerator
from benchmark.storage import StorageManager


@pytest.fixture
def storage(tmp_path: Path) -> StorageManager:
    return StorageManager(tmp_path)


@pytest.fixture
def reporter(storage: StorageManager) -> ReportGenerator:
    return ReportGenerator(storage)


def _seed_run(storage: StorageManager, run_id: str, results: list[dict], scores: list[dict] | None = None):
    """Helper: create a run directory with results and optional scores."""
    storage.create_run_dir(run_id)
    for r in results:
        storage.append_result(run_id, r)
    if scores is not None:
        scores_path = storage.runs_dir / run_id / "scores.json"
        storage.write_json(scores_path, {"scores": scores})


def _make_result(prompt_id: str, model_name: str, category: str,
                 total_time_ms: float = 1000.0, success: bool = True,
                 vram_used_mb: float | None = None) -> dict:
    result: dict = {
        "prompt_id": prompt_id,
        "model_name": model_name,
        "category": category,
        "response_text": "test response",
        "success": success,
        "error": None,
        "metrics": {
            "total_time_ms": total_time_ms,
            "tokens_per_second": 10.0,
            "ttft_ms": 50.0,
            "prompt_tokens": 10,
            "completion_tokens": 20,
        },
    }
    if vram_used_mb is not None:
        result["vram_snapshot"] = {
            "timestamp": "2025-01-01T00:00:00",
            "used_mb": vram_used_mb,
            "total_mb": 16384.0,
            "source": "nvidia_smi",
        }
    return result


# --- generate_matrix ---

class TestGenerateMatrix:
    def test_basic_matrix(self, storage, reporter):
        results = [
            _make_result("p1", "modelA", "coding", total_time_ms=100),
            _make_result("p2", "modelA", "coding", total_time_ms=200),
            _make_result("p1", "modelB", "coding", total_time_ms=300),
        ]
        scores = [
            {"prompt_id": "p1", "model_name": "modelA", "manual_score": 8, "judge_score": None, "comment": None},
            {"prompt_id": "p2", "model_name": "modelA", "manual_score": 6, "judge_score": None, "comment": None},
            {"prompt_id": "p1", "model_name": "modelB", "manual_score": 9, "judge_score": None, "comment": None},
        ]
        _seed_run(storage, "run1", results, scores)

        matrix = reporter.generate_matrix("run1")
        assert "modelA" in matrix.models
        assert "modelB" in matrix.models
        assert "coding" in matrix.categories

        cell_a = matrix.cells["modelA"]["coding"]
        assert cell_a.avg_score == pytest.approx(7.0)  # (8+6)/2
        assert cell_a.avg_time_ms == pytest.approx(150.0)  # (100+200)/2
        assert cell_a.sample_count == 2

        cell_b = matrix.cells["modelB"]["coding"]
        assert cell_b.avg_score == pytest.approx(9.0)
        assert cell_b.sample_count == 1

    def test_empty_results(self, storage, reporter):
        _seed_run(storage, "run_empty", [])
        matrix = reporter.generate_matrix("run_empty")
        assert matrix.models == []
        assert matrix.categories == []
        assert matrix.cells == {}

    def test_no_scores(self, storage, reporter):
        results = [_make_result("p1", "modelA", "translation")]
        _seed_run(storage, "run_no_scores", results)
        matrix = reporter.generate_matrix("run_no_scores")
        cell = matrix.cells["modelA"]["translation"]
        assert cell.avg_score is None
        assert cell.avg_time_ms is not None
        assert cell.sample_count == 1

    def test_judge_score_fallback(self, storage, reporter):
        results = [_make_result("p1", "modelA", "coding")]
        scores = [
            {"prompt_id": "p1", "model_name": "modelA", "manual_score": None, "judge_score": 7.5, "comment": None},
        ]
        _seed_run(storage, "run_judge", results, scores)
        matrix = reporter.generate_matrix("run_judge")
        assert matrix.cells["modelA"]["coding"].avg_score == pytest.approx(7.5)

    def test_manual_score_preferred_over_judge(self, storage, reporter):
        results = [_make_result("p1", "modelA", "coding")]
        scores = [
            {"prompt_id": "p1", "model_name": "modelA", "manual_score": 9, "judge_score": 5.0, "comment": None},
        ]
        _seed_run(storage, "run_prefer", results, scores)
        matrix = reporter.generate_matrix("run_prefer")
        assert matrix.cells["modelA"]["coding"].avg_score == pytest.approx(9.0)


# --- generate_ranking ---

class TestGenerateRanking:
    def test_ranking_order(self, storage, reporter):
        results = [
            _make_result("p1", "modelA", "coding", total_time_ms=100),
            _make_result("p1", "modelB", "coding", total_time_ms=200),
            _make_result("p1", "modelC", "coding", total_time_ms=300),
        ]
        scores = [
            {"prompt_id": "p1", "model_name": "modelA", "manual_score": 5, "judge_score": None, "comment": None},
            {"prompt_id": "p1", "model_name": "modelB", "manual_score": 9, "judge_score": None, "comment": None},
            {"prompt_id": "p1", "model_name": "modelC", "manual_score": 7, "judge_score": None, "comment": None},
        ]
        _seed_run(storage, "run_rank", results, scores)

        ranking = reporter.generate_ranking("run_rank")
        assert len(ranking) == 3
        assert ranking[0].model_name == "modelB"
        assert ranking[0].rank == 1
        assert ranking[1].model_name == "modelC"
        assert ranking[2].model_name == "modelA"

    def test_ranking_with_category_filter(self, storage, reporter):
        results = [
            _make_result("p1", "modelA", "coding"),
            _make_result("p2", "modelA", "translation"),
            _make_result("p1", "modelB", "coding"),
        ]
        scores = [
            {"prompt_id": "p1", "model_name": "modelA", "manual_score": 5, "judge_score": None, "comment": None},
            {"prompt_id": "p2", "model_name": "modelA", "manual_score": 10, "judge_score": None, "comment": None},
            {"prompt_id": "p1", "model_name": "modelB", "manual_score": 8, "judge_score": None, "comment": None},
        ]
        _seed_run(storage, "run_cat", results, scores)

        ranking = reporter.generate_ranking("run_cat", category="coding")
        assert len(ranking) == 2
        assert ranking[0].model_name == "modelB"
        assert ranking[0].avg_score == pytest.approx(8.0)

    def test_ranking_empty(self, storage, reporter):
        _seed_run(storage, "run_empty2", [])
        ranking = reporter.generate_ranking("run_empty2")
        assert ranking == []


# --- generate_vram_report ---

class TestGenerateVRAMReport:
    def test_vram_stats(self, storage, reporter):
        results = [
            _make_result("p1", "modelA", "coding", vram_used_mb=8000.0),
            _make_result("p2", "modelA", "coding", vram_used_mb=10000.0),
            _make_result("p1", "modelB", "coding", vram_used_mb=14000.0),
        ]
        _seed_run(storage, "run_vram", results)

        report = reporter.generate_vram_report("run_vram")
        assert len(report.models_within_limit) == 2  # no limit, all go to within
        assert report.vram_limit_mb is None

    def test_vram_limit_partitioning(self, storage, reporter):
        results = [
            _make_result("p1", "modelA", "coding", vram_used_mb=8000.0),
            _make_result("p2", "modelA", "coding", vram_used_mb=10000.0),
            _make_result("p1", "modelB", "coding", vram_used_mb=20000.0),
        ]
        _seed_run(storage, "run_vram_limit", results)

        report = reporter.generate_vram_report("run_vram_limit", vram_limit_mb=16384)
        within_names = {m["name"] for m in report.models_within_limit}
        exceeding_names = {m["name"] for m in report.models_exceeding_limit}
        assert "modelA" in within_names  # peak 10000 <= 16384
        assert "modelB" in exceeding_names  # peak 20000 > 16384

    def test_vram_no_snapshots(self, storage, reporter):
        results = [_make_result("p1", "modelA", "coding")]
        _seed_run(storage, "run_no_vram", results)
        report = reporter.generate_vram_report("run_no_vram")
        assert report.models_within_limit == []
        assert report.models_exceeding_limit == []


# --- export_json ---

class TestExportJSON:
    def test_export_json(self, storage, reporter, tmp_path):
        results = [_make_result("p1", "modelA", "coding")]
        scores = [
            {"prompt_id": "p1", "model_name": "modelA", "manual_score": 8, "judge_score": None, "comment": None},
        ]
        _seed_run(storage, "run_json", results, scores)

        out = tmp_path / "export.json"
        reporter.export_json("run_json", out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["run_id"] == "run_json"
        assert len(data["results"]) == 1
        assert len(data["scores"]) == 1


# --- export_csv ---

class TestExportCSV:
    def test_export_csv_basic(self, storage, reporter, tmp_path):
        results = [_make_result("p1", "modelA", "translation")]
        _seed_run(storage, "run_csv", results)

        out = tmp_path / "export.csv"
        reporter.export_csv("run_csv", out)

        content = out.read_text(encoding="utf-8-sig")
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "prompt_id" in lines[0]
        assert "translation" in lines[1]  # Turkish chars preserved

    def test_export_csv_turkish_chars(self, storage, reporter, tmp_path):
        results = [_make_result("p1", "modelA", "creative")]
        _seed_run(storage, "run_csv_tr", results)

        out = tmp_path / "export_tr.csv"
        reporter.export_csv("run_csv_tr", out)

        raw = out.read_bytes()
        assert raw[:3] == b"\xef\xbb\xbf"  # UTF-8 BOM
        content = raw.decode("utf-8-sig")
        assert "creative" in content


# --- generate_summary ---

class TestGenerateSummary:
    def test_summary(self, storage, reporter):
        results = [
            _make_result("p1", "modelA", "coding", success=True),
            _make_result("p2", "modelA", "coding", success=True),
            _make_result("p1", "modelB", "coding", success=False),
        ]
        _seed_run(storage, "run_summary", results)

        summary = reporter.generate_summary("run_summary")
        assert summary.run_id == "run_summary"
        assert summary.total_prompts == 2  # p1, p2
        assert summary.total_models == 2  # modelA, modelB
        assert summary.successful_results == 2
        assert summary.failed_results == 1
        assert summary.matrix is not None

    def test_summary_with_meta(self, storage, reporter):
        results = [_make_result("p1", "modelA", "coding")]
        _seed_run(storage, "run_meta", results)
        meta_path = storage.runs_dir / "run_meta" / "meta.json"
        storage.write_json(meta_path, {"timestamp": "2025-06-15T10:30:00"})

        summary = reporter.generate_summary("run_meta")
        assert summary.timestamp.year == 2025
        assert summary.timestamp.month == 6
