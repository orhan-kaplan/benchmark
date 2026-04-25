# Feature: colab-benchmark-system, Property 13: Karşılaştırma Matrisi Doğruluğu
# Feature: colab-benchmark-system, Property 14: Sıralama Düzeni İnvariantı
# Feature: colab-benchmark-system, Property 15: CSV Dışa Aktarma Veri Bütünlüğü
# Feature: colab-benchmark-system, Property 17: VRAM Limit Bölümleme
# Validates: Requirements 6.1, 6.2, 6.4, 6.6, 7.4, 12.2, 12.4
"""
Property-based tests for ReportGenerator.

Property 13: For any result set, the comparison matrix's cells[model][category].avg_score
should equal the arithmetic mean of all scores for that model-category pair; and avg_time_ms
should equal the mean of response times.

Property 14: For any result set and any optional category filter, the ranking list should be
sorted by avg_score in descending order; and when a category filter is applied, only data from
that category should be used.

Property 15: For any result set, the CSV export should have: (a) row count equal to result
count (excluding header), (b) all required columns present, (c) Turkish characters
(ç, ğ, ı, ö, ş, ü) preserved.

Property 17: For any set of models with peak VRAM values and any VRAM limit,
models_within_limit should all have peak_mb <= limit, models_exceeding_limit should all
have peak_mb > limit, and the union of both lists should cover all models.
"""

import csv
import io
from collections import defaultdict

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from benchmark.reporter import ReportGenerator
from benchmark.storage import StorageManager


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

model_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=1,
    max_size=15,
)

category_st = st.sampled_from(["coding", "translation", "reasoning", "creative", "general"])

score_st = st.integers(min_value=1, max_value=10)

time_ms_st = st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False)

vram_mb_st = st.floats(min_value=100.0, max_value=50_000.0, allow_nan=False, allow_infinity=False)


def _make_result(prompt_id: str, model_name: str, category: str,
                 total_time_ms: float = 1000.0,
                 vram_used_mb: float | None = None) -> dict:
    """Build a result dict matching the JSONL schema."""
    result: dict = {
        "prompt_id": prompt_id,
        "model_name": model_name,
        "category": category,
        "response_text": "yanıt metni — çğıöşü",
        "success": True,
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


def _seed_run(storage: StorageManager, run_id: str,
              results: list[dict], scores: list[dict] | None = None):
    """Seed a run directory with results and optional scores."""
    storage.create_run_dir(run_id)
    for r in results:
        storage.append_result(run_id, r)
    if scores is not None:
        scores_path = storage.runs_dir / run_id / "scores.json"
        storage.write_json(scores_path, {"scores": scores})


# ---------------------------------------------------------------------------
# Composite strategy: a full run with results + scores
# ---------------------------------------------------------------------------

@st.composite
def scored_run_st(draw):
    """Generate a list of (result_dict, score_dict) pairs with 1-5 models and 1-5 categories."""
    num_models = draw(st.integers(min_value=1, max_value=4))
    num_categories = draw(st.integers(min_value=1, max_value=3))
    prompts_per_pair = draw(st.integers(min_value=1, max_value=3))

    models = [f"model_{i}" for i in range(num_models)]
    categories = draw(
        st.lists(category_st, min_size=num_categories, max_size=num_categories, unique=True)
    )

    results = []
    scores = []
    pid_counter = 0

    for cat in categories:
        for _ in range(prompts_per_pair):
            pid = f"p{pid_counter}"
            pid_counter += 1
            for model in models:
                t = draw(time_ms_st)
                s = draw(score_st)
                results.append(_make_result(pid, model, cat, total_time_ms=t))
                scores.append({
                    "prompt_id": pid,
                    "model_name": model,
                    "manual_score": s,
                    "judge_score": None,
                    "comment": None,
                })

    return results, scores, models, categories


# ---------------------------------------------------------------------------
# Property 13: Karşılaştırma Matrisi Doğruluğu
# **Validates: Requirements 6.1**
# ---------------------------------------------------------------------------


@given(data=scored_run_st())
@settings(max_examples=100)
def test_comparison_matrix_accuracy(data, tmp_path_factory):
    """Each cell's avg_score and avg_time_ms equal the arithmetic mean of the
    corresponding model-category pair values."""
    results, scores, models, categories = data

    base = tmp_path_factory.mktemp("matrix")
    storage = StorageManager(base)
    reporter = ReportGenerator(storage)

    run_id = "run_matrix"
    _seed_run(storage, run_id, results, scores)

    matrix = reporter.generate_matrix(run_id)

    # Build expected values manually
    expected_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    expected_times: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    score_lookup = {(s["prompt_id"], s["model_name"]): s for s in scores}

    for r in results:
        model = r["model_name"]
        cat = r["category"]
        key = (r["prompt_id"], model)
        se = score_lookup.get(key)
        if se and se.get("manual_score") is not None:
            expected_scores[model][cat].append(float(se["manual_score"]))
        t = r.get("metrics", {}).get("total_time_ms")
        if t is not None:
            expected_times[model][cat].append(t)

    for model in models:
        for cat in categories:
            cell = matrix.cells[model][cat]

            # avg_score
            s_list = expected_scores[model][cat]
            if s_list:
                expected_avg = sum(s_list) / len(s_list)
                assert cell.avg_score == pytest.approx(expected_avg, rel=1e-6)
            else:
                assert cell.avg_score is None

            # avg_time_ms
            t_list = expected_times[model][cat]
            if t_list:
                expected_avg_t = sum(t_list) / len(t_list)
                assert cell.avg_time_ms == pytest.approx(expected_avg_t, rel=1e-6)
            else:
                assert cell.avg_time_ms is None


# ---------------------------------------------------------------------------
# Property 14: Sıralama Düzeni İnvariantı
# **Validates: Requirements 6.2, 6.6**
# ---------------------------------------------------------------------------


@given(data=scored_run_st())
@settings(max_examples=100)
def test_ranking_descending_order(data, tmp_path_factory):
    """The ranking list is sorted by avg_score in descending order."""
    results, scores, _models, _categories = data

    base = tmp_path_factory.mktemp("rank")
    storage = StorageManager(base)
    reporter = ReportGenerator(storage)

    run_id = "run_rank"
    _seed_run(storage, run_id, results, scores)

    ranking = reporter.generate_ranking(run_id)

    # Verify descending order (None values go last)
    scored = [r for r in ranking if r.avg_score is not None]
    for i in range(len(scored) - 1):
        assert scored[i].avg_score >= scored[i + 1].avg_score


@given(data=scored_run_st())
@settings(max_examples=100)
def test_ranking_category_filter_uses_only_that_category(data, tmp_path_factory):
    """When a category filter is applied, only data from that category is used."""
    results, scores, models, categories = data
    assume(len(categories) >= 1)

    base = tmp_path_factory.mktemp("rank_cat")
    storage = StorageManager(base)
    reporter = ReportGenerator(storage)

    run_id = "run_rank_cat"
    _seed_run(storage, run_id, results, scores)

    target_cat = categories[0]
    ranking = reporter.generate_ranking(run_id, category=target_cat)

    # Manually compute expected avg_score per model for the target category
    score_lookup = {(s["prompt_id"], s["model_name"]): s for s in scores}
    expected: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r["category"] != target_cat:
            continue
        key = (r["prompt_id"], r["model_name"])
        se = score_lookup.get(key)
        if se and se.get("manual_score") is not None:
            expected[r["model_name"]].append(float(se["manual_score"]))

    for entry in ranking:
        if entry.avg_score is not None:
            s_list = expected[entry.model_name]
            assert len(s_list) > 0
            assert entry.avg_score == pytest.approx(sum(s_list) / len(s_list), rel=1e-6)


# ---------------------------------------------------------------------------
# Property 15: CSV Dışa Aktarma Veri Bütünlüğü
# **Validates: Requirements 6.4, 12.2, 12.4**
# ---------------------------------------------------------------------------

REQUIRED_CSV_COLUMNS = {
    "prompt_id", "model_name", "category", "success", "response_text",
    "total_time_ms", "tokens_per_second", "ttft_ms", "prompt_tokens",
    "completion_tokens", "vram_used_mb", "manual_score", "judge_score", "error",
}

TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")


@st.composite
def csv_run_st(draw):
    """Generate results that include Turkish characters in response_text."""
    num = draw(st.integers(min_value=1, max_value=8))
    turkish_snippets = [
        "çeviri güzel",
        "öğrenci başarılı",
        "şüphesiz doğru",
        "ığdır şehri",
    ]
    results = []
    for i in range(num):
        model = f"model_{draw(st.integers(min_value=0, max_value=2))}"
        cat = draw(category_st)
        snippet = draw(st.sampled_from(turkish_snippets))
        r = _make_result(f"p{i}", model, cat)
        r["response_text"] = snippet
        results.append(r)
    return results


@given(results=csv_run_st())
@settings(max_examples=100)
def test_csv_export_integrity(results, tmp_path_factory):
    """CSV export has correct row count, all required columns, and preserves Turkish chars."""
    base = tmp_path_factory.mktemp("csv")
    storage = StorageManager(base)
    reporter = ReportGenerator(storage)

    run_id = "run_csv"
    _seed_run(storage, run_id, results)

    out = base / "export.csv"
    reporter.export_csv(run_id, out)

    raw = out.read_bytes()
    # UTF-8 BOM check
    assert raw[:3] == b"\xef\xbb\xbf", "CSV must start with UTF-8 BOM"

    content = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)

    # (a) Row count == result count
    assert len(rows) == len(results)

    # (b) All required columns present
    assert REQUIRED_CSV_COLUMNS.issubset(set(reader.fieldnames or []))

    # (c) Turkish characters preserved
    all_text = content
    # Verify at least some Turkish chars appear in the CSV
    found_turkish = any(ch in all_text for ch in TURKISH_CHARS)
    assert found_turkish, "Turkish characters must be preserved in CSV output"


# ---------------------------------------------------------------------------
# Property 17: VRAM Limit Bölümleme
# **Validates: Requirements 7.4**
# ---------------------------------------------------------------------------


@st.composite
def vram_run_st(draw):
    """Generate results with VRAM snapshots for multiple models."""
    num_models = draw(st.integers(min_value=1, max_value=5))
    prompts_per_model = draw(st.integers(min_value=1, max_value=3))

    results = []
    model_peak: dict[str, float] = {}
    pid = 0

    for m_idx in range(num_models):
        model = f"model_{m_idx}"
        peak = 0.0
        for _ in range(prompts_per_model):
            vram = draw(vram_mb_st)
            peak = max(peak, vram)
            results.append(
                _make_result(f"p{pid}", model, "coding", vram_used_mb=vram)
            )
            pid += 1
        model_peak[model] = peak

    vram_limit = draw(st.integers(min_value=1000, max_value=40_000))
    return results, model_peak, vram_limit


@given(data=vram_run_st())
@settings(max_examples=100)
def test_vram_limit_partitioning(data, tmp_path_factory):
    """models_within_limit all have peak_mb <= limit, models_exceeding_limit all
    have peak_mb > limit, and the union covers all models."""
    results, model_peak, vram_limit = data

    base = tmp_path_factory.mktemp("vram")
    storage = StorageManager(base)
    reporter = ReportGenerator(storage)

    run_id = "run_vram"
    _seed_run(storage, run_id, results)

    report = reporter.generate_vram_report(run_id, vram_limit_mb=vram_limit)

    within_names = {m["name"] for m in report.models_within_limit}
    exceeding_names = {m["name"] for m in report.models_exceeding_limit}

    # All within-limit models have peak_mb <= limit
    for m in report.models_within_limit:
        assert m["peak_mb"] <= vram_limit, (
            f"{m['name']} peak {m['peak_mb']} should be <= {vram_limit}"
        )

    # All exceeding-limit models have peak_mb > limit
    for m in report.models_exceeding_limit:
        assert m["peak_mb"] > vram_limit, (
            f"{m['name']} peak {m['peak_mb']} should be > {vram_limit}"
        )

    # No overlap
    assert within_names.isdisjoint(exceeding_names)

    # Union covers all models that have VRAM data
    assert within_names | exceeding_names == set(model_peak.keys())
