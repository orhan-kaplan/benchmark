"""Rapor üretici — karşılaştırma matrisi, sıralama ve dışa aktarma."""

import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from benchmark.models import (
    ComparisonMatrix,
    MatrixCell,
    ModelRanking,
    RunSummary,
    VRAMReport,
)
from benchmark.storage import StorageManager

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Benchmark sonuçlarından karşılaştırma matrisi ve raporlar üreten sınıf.

    StorageManager üzerinden sonuçları ve puanları okur, model × kategori
    matrisi, sıralama, VRAM raporu ve özet rapor üretir. JSON ve CSV
    formatlarında dışa aktarma destekler.
    """

    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage

    # -- internal helpers --

    def _load_scores_lookup(self, run_id: str) -> dict[tuple[str, str], dict]:
        """Load scores.json and build a (prompt_id, model_name) -> score dict lookup."""
        scores_path = self.storage.runs_dir / run_id / "scores.json"
        if not scores_path.exists():
            return {}
        data = self.storage.read_json(scores_path)
        lookup: dict[tuple[str, str], dict] = {}
        for s in data.get("scores", []):
            key = (s["prompt_id"], s["model_name"])
            lookup[key] = s
        return lookup

    def _get_score_value(self, score_entry: dict) -> Optional[float]:
        """Extract the best available score: prefer manual_score, fallback to judge_score."""
        manual = score_entry.get("manual_score")
        if manual is not None:
            return float(manual)
        judge = score_entry.get("judge_score")
        if judge is not None:
            return float(judge)
        return None

    # -- public API --

    def generate_matrix(self, run_id: str) -> ComparisonMatrix:
        """Model × kategori karşılaştırma matrisi üret.

        Her hücrede ortalama puan (avg_score), ortalama yanıt süresi
        (avg_time_ms) ve örnek sayısı (sample_count) bulunur.

        Puanlar scores.json'dan okunur (manual_score tercih edilir,
        yoksa judge_score kullanılır). Kategori bilgisi results.jsonl
        içindeki her sonucun ``category`` alanından alınır.

        Args:
            run_id: Çalıştırma kimliği.

        Returns:
            ComparisonMatrix nesnesi.
        """
        results = self.storage.read_results(run_id)
        scores_lookup = self._load_scores_lookup(run_id)

        # Collect data per (model, category)
        # Each entry: {"scores": [...], "times": [...]}
        cell_data: dict[str, dict[str, dict]] = defaultdict(
            lambda: defaultdict(lambda: {"scores": [], "times": [], "count": 0})
        )
        models_set: set[str] = set()
        categories_set: set[str] = set()

        for result in results:
            model_name = result.get("model_name", "")
            category = result.get("category", "unknown")
            models_set.add(model_name)
            categories_set.add(category)

            data = cell_data[model_name][category]
            data["count"] += 1

            # Score from scores.json
            key = (result.get("prompt_id", ""), model_name)
            score_entry = scores_lookup.get(key)
            if score_entry is not None:
                score_val = self._get_score_value(score_entry)
                if score_val is not None:
                    data["scores"].append(score_val)

            # Time from metrics
            metrics = result.get("metrics")
            if metrics and metrics.get("total_time_ms") is not None:
                data["times"].append(metrics["total_time_ms"])

        models = sorted(models_set)
        categories = sorted(categories_set)

        # Build cells dict
        cells: dict[str, dict[str, MatrixCell]] = {}
        for model in models:
            cells[model] = {}
            for cat in categories:
                d = cell_data[model][cat]
                avg_score = (
                    sum(d["scores"]) / len(d["scores"]) if d["scores"] else None
                )
                avg_time = (
                    sum(d["times"]) / len(d["times"]) if d["times"] else None
                )
                cells[model][cat] = MatrixCell(
                    avg_score=avg_score,
                    avg_time_ms=avg_time,
                    sample_count=d["count"],
                )

        return ComparisonMatrix(models=models, categories=categories, cells=cells)

    def generate_ranking(
        self, run_id: str, category: Optional[str] = None
    ) -> list[ModelRanking]:
        """Model sıralaması üret (avg_score'a göre azalan).

        Args:
            run_id: Çalıştırma kimliği.
            category: Opsiyonel kategori filtresi. Belirtilirse yalnızca
                o kategoriye ait veriler kullanılır.

        Returns:
            Sıralı ModelRanking listesi.
        """
        results = self.storage.read_results(run_id)
        scores_lookup = self._load_scores_lookup(run_id)

        # Accumulate per model
        model_data: dict[str, dict] = defaultdict(
            lambda: {"scores": [], "times": []}
        )

        for result in results:
            # Apply category filter
            if category is not None and result.get("category") != category:
                continue

            model_name = result.get("model_name", "")

            key = (result.get("prompt_id", ""), model_name)
            score_entry = scores_lookup.get(key)
            if score_entry is not None:
                score_val = self._get_score_value(score_entry)
                if score_val is not None:
                    model_data[model_name]["scores"].append(score_val)

            metrics = result.get("metrics")
            if metrics and metrics.get("total_time_ms") is not None:
                model_data[model_name]["times"].append(metrics["total_time_ms"])

        # Build ranking entries
        ranking_entries: list[dict] = []
        for model_name, data in model_data.items():
            avg_score = (
                sum(data["scores"]) / len(data["scores"]) if data["scores"] else None
            )
            avg_time = (
                sum(data["times"]) / len(data["times"]) if data["times"] else None
            )
            ranking_entries.append(
                {"model_name": model_name, "avg_score": avg_score, "avg_time_ms": avg_time}
            )

        # Sort by avg_score descending (None values go last)
        ranking_entries.sort(
            key=lambda x: (x["avg_score"] is not None, x["avg_score"] or 0),
            reverse=True,
        )

        # Assign ranks
        rankings: list[ModelRanking] = []
        for i, entry in enumerate(ranking_entries, start=1):
            rankings.append(
                ModelRanking(
                    model_name=entry["model_name"],
                    rank=i,
                    avg_score=entry["avg_score"],
                    avg_time_ms=entry["avg_time_ms"],
                )
            )

        return rankings

    def generate_vram_report(
        self, run_id: str, vram_limit_mb: Optional[int] = None
    ) -> VRAMReport:
        """VRAM kullanım raporu üret.

        Her model için peak_mb ve avg_mb hesaplar. Eğer vram_limit_mb
        belirtilmişse modelleri limite göre ikiye ayırır.

        Args:
            run_id: Çalıştırma kimliği.
            vram_limit_mb: Opsiyonel VRAM limiti (MB).

        Returns:
            VRAMReport nesnesi.
        """
        results = self.storage.read_results(run_id)

        # Collect VRAM measurements per model
        model_vram: dict[str, list[float]] = defaultdict(list)
        for result in results:
            vram = result.get("vram_snapshot")
            if vram and vram.get("used_mb") is not None:
                model_vram[result.get("model_name", "")].append(vram["used_mb"])

        # Compute stats per model
        within: list[dict] = []
        exceeding: list[dict] = []

        for model_name, measurements in model_vram.items():
            peak_mb = max(measurements)
            avg_mb = sum(measurements) / len(measurements)
            entry = {"name": model_name, "peak_mb": peak_mb, "avg_mb": avg_mb}

            if vram_limit_mb is not None:
                if peak_mb <= vram_limit_mb:
                    within.append(entry)
                else:
                    exceeding.append(entry)
            else:
                within.append(entry)

        return VRAMReport(
            models_within_limit=within,
            models_exceeding_limit=exceeding,
            vram_limit_mb=vram_limit_mb,
        )

    def export_json(self, run_id: str, output_path: Path) -> None:
        """Benchmark sonuçlarını JSON formatında dışa aktar.

        Sonuçlar ve puanlar tek bir JSON dosyasında birleştirilir.

        Args:
            run_id: Çalıştırma kimliği.
            output_path: Hedef dosya yolu.
        """
        results = self.storage.read_results(run_id)
        scores_lookup = self._load_scores_lookup(run_id)

        export_data = {
            "run_id": run_id,
            "results": results,
            "scores": list(scores_lookup.values()),
        }

        self.storage.write_json(Path(output_path), export_data)

    def export_csv(self, run_id: str, output_path: Path) -> None:
        """Benchmark sonuçlarını CSV formatında dışa aktar (UTF-8 BOM).

        Args:
            run_id: Çalıştırma kimliği.
            output_path: Hedef dosya yolu.
        """
        results = self.storage.read_results(run_id)
        scores_lookup = self._load_scores_lookup(run_id)

        rows: list[dict] = []
        for result in results:
            key = (result.get("prompt_id", ""), result.get("model_name", ""))
            score_entry = scores_lookup.get(key, {})

            metrics = result.get("metrics") or {}
            vram = result.get("vram_snapshot") or {}

            row = {
                "prompt_id": result.get("prompt_id", ""),
                "model_name": result.get("model_name", ""),
                "category": result.get("category", ""),
                "success": result.get("success", ""),
                "response_text": result.get("response_text", ""),
                "total_time_ms": metrics.get("total_time_ms", ""),
                "tokens_per_second": metrics.get("tokens_per_second", ""),
                "ttft_ms": metrics.get("ttft_ms", ""),
                "prompt_tokens": metrics.get("prompt_tokens", ""),
                "completion_tokens": metrics.get("completion_tokens", ""),
                "vram_used_mb": vram.get("used_mb", ""),
                "manual_score": score_entry.get("manual_score", ""),
                "judge_score": score_entry.get("judge_score", ""),
                "error": result.get("error", ""),
            }
            rows.append(row)

        fieldnames = [
            "prompt_id",
            "model_name",
            "category",
            "success",
            "response_text",
            "total_time_ms",
            "tokens_per_second",
            "ttft_ms",
            "prompt_tokens",
            "completion_tokens",
            "vram_used_mb",
            "manual_score",
            "judge_score",
            "error",
        ]

        self.storage.write_csv(Path(output_path), rows, fieldnames)

    def generate_summary(self, run_id: str) -> RunSummary:
        """Çalıştırma özet raporu üret.

        Args:
            run_id: Çalıştırma kimliği.

        Returns:
            RunSummary nesnesi.
        """
        results = self.storage.read_results(run_id)
        matrix = self.generate_matrix(run_id)

        # Try to load timestamp from meta.json
        meta_path = self.storage.runs_dir / run_id / "meta.json"
        if meta_path.exists():
            meta = self.storage.read_json(meta_path)
            timestamp = datetime.fromisoformat(meta.get("timestamp", datetime.now(timezone.utc).isoformat()))
        else:
            timestamp = datetime.now(timezone.utc)

        models_set: set[str] = set()
        prompt_ids: set[str] = set()
        successful = 0
        failed = 0

        for result in results:
            models_set.add(result.get("model_name", ""))
            prompt_ids.add(result.get("prompt_id", ""))
            if result.get("success", True):
                successful += 1
            else:
                failed += 1

        return RunSummary(
            run_id=run_id,
            timestamp=timestamp,
            total_prompts=len(prompt_ids),
            total_models=len(models_set),
            successful_results=successful,
            failed_results=failed,
            matrix=matrix,
        )
