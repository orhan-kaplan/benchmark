"""Puanlama motoru — manuel ve otomatik (judge model) puanlama."""

import logging
import re
from typing import Optional

from benchmark.api_client import APIClient
from benchmark.models import (
    GenerationParams,
    JudgeConfig,
    ScoreEntry,
)
from benchmark.storage import StorageManager

logger = logging.getLogger(__name__)


class ScoringEngine:
    """Manuel ve yargıcı model (judge) puanlamasını yöneten sınıf.

    Puanları ``scores.json`` dosyasında saklar. Manuel puanlar doğrudan
    girilir; otomatik puanlar bir judge model aracılığıyla API Client
    üzerinden alınır.
    """

    def __init__(
        self,
        storage: StorageManager,
        api_client: Optional[APIClient] = None,
    ) -> None:
        self.storage = storage
        self.api_client = api_client

    # -- helpers --

    def _scores_path(self, run_id: str):
        """Return the Path to scores.json for a given run."""
        return self.storage.runs_dir / run_id / "scores.json"

    def _load_scores(self, run_id: str) -> list[dict]:
        """Load existing scores from scores.json, or return empty list."""
        path = self._scores_path(run_id)
        if not path.exists():
            return []
        data = self.storage.read_json(path)
        return data.get("scores", [])

    def _save_scores(self, run_id: str, scores: list[dict]) -> None:
        """Persist scores list to scores.json."""
        path = self._scores_path(run_id)
        self.storage.write_json(path, {"scores": scores})

    # -- public API --

    def add_manual_score(
        self,
        run_id: str,
        prompt_id: str,
        model_name: str,
        score: int,
        comment: Optional[str] = None,
    ) -> None:
        """Add or update a manual score for a prompt-model pair.

        Args:
            run_id: Benchmark run identifier.
            prompt_id: Prompt identifier.
            model_name: Model name.
            score: Integer score between 1 and 10 (inclusive).
            comment: Optional text comment.

        Raises:
            ValueError: If score is outside the 1-10 range.
        """
        if score < 1 or score > 10:
            raise ValueError(f"Score must be between 1 and 10, got {score}")

        scores = self._load_scores(run_id)

        # Find existing entry or create new one
        found = False
        for entry in scores:
            if entry["prompt_id"] == prompt_id and entry["model_name"] == model_name:
                entry["manual_score"] = score
                entry["comment"] = comment
                found = True
                break

        if not found:
            scores.append(
                {
                    "prompt_id": prompt_id,
                    "model_name": model_name,
                    "manual_score": score,
                    "judge_score": None,
                    "comment": comment,
                }
            )

        self._save_scores(run_id, scores)

    async def auto_score(self, run_id: str, judge_config: JudgeConfig) -> None:
        """Automatically score all unscored results using a judge model.

        For each unscored result, builds a judge prompt from the
        ``JudgeConfig`` templates, sends it to the judge model via
        ``api_client``, parses the numeric score from the response,
        and saves it.

        Args:
            run_id: Benchmark run identifier.
            judge_config: Judge model configuration with templates.

        Raises:
            RuntimeError: If no api_client was provided.
        """
        if self.api_client is None:
            raise RuntimeError("api_client is required for auto_score")

        unscored = self.get_unscored(run_id)
        if not unscored:
            logger.info("No unscored items for run %s", run_id)
            return

        scores = self._load_scores(run_id)

        for item in unscored:
            prompt_id = item["prompt_id"]
            model_name = item["model_name"]
            question = item.get("prompt_text", "")
            answer = item.get("response_text", "")

            # Pick the right template
            category = item.get("category", "default")
            template = judge_config.prompt_templates.get(
                category,
                judge_config.prompt_templates.get("default", ""),
            )
            judge_prompt = template.format(question=question, answer=answer)

            # Call judge model
            try:
                response = await self.api_client.chat_completion(
                    endpoint=judge_config.api_endpoint,
                    model=judge_config.model_name,
                    messages=[{"role": "user", "content": judge_prompt}],
                    params=GenerationParams(temperature=0.0, max_tokens=64),
                )

                judge_score = self._parse_score(response.response_text)
            except Exception:
                logger.exception(
                    "Failed to auto-score prompt_id=%s model=%s",
                    prompt_id,
                    model_name,
                )
                continue

            # Update or add score entry
            found = False
            for entry in scores:
                if entry["prompt_id"] == prompt_id and entry["model_name"] == model_name:
                    entry["judge_score"] = judge_score
                    found = True
                    break

            if not found:
                scores.append(
                    {
                        "prompt_id": prompt_id,
                        "model_name": model_name,
                        "manual_score": None,
                        "judge_score": judge_score,
                        "comment": None,
                    }
                )

        self._save_scores(run_id, scores)

    def get_unscored(self, run_id: str) -> list[dict]:
        """Return results that have no manual or judge score.

        Loads results from JSONL and scores from scores.json, then
        returns items where both ``manual_score`` and ``judge_score``
        are ``None`` (or the item has no score entry at all).

        Returns:
            List of dicts with prompt_id, model_name, and result data.
        """
        results = self.storage.read_results(run_id)
        scores = self._load_scores(run_id)

        # Build a lookup of scored items
        scored_lookup: dict[tuple[str, str], dict] = {}
        for s in scores:
            key = (s["prompt_id"], s["model_name"])
            scored_lookup[key] = s

        unscored: list[dict] = []
        for result in results:
            key = (result["prompt_id"], result["model_name"])
            score_entry = scored_lookup.get(key)

            if score_entry is None:
                # No score entry at all — unscored
                unscored.append(result)
            elif score_entry.get("manual_score") is None and score_entry.get("judge_score") is None:
                # Entry exists but both scores are None — still unscored
                unscored.append(result)

        return unscored

    def get_scores(self, run_id: str) -> list[ScoreEntry]:
        """Load and return all ScoreEntry objects for a run.

        Returns:
            List of ScoreEntry Pydantic models.
        """
        raw_scores = self._load_scores(run_id)
        return [ScoreEntry(**s) for s in raw_scores]

    # -- internal --

    @staticmethod
    def _parse_score(text: Optional[str]) -> Optional[float]:
        """Extract a numeric score from judge model response text.

        Looks for the first number (int or float) in the text.
        Returns None if no number is found or text is None.
        """
        if not text:
            return None
        match = re.search(r"(\d+(?:\.\d+)?)", text.strip())
        if match:
            value = float(match.group(1))
            # Clamp to 1-10 range
            return max(1.0, min(10.0, value))
        return None
