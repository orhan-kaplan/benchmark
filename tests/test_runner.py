"""BenchmarkRunner birim testleri — mock bağımlılıklarla."""

import asyncio
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmark.api_client import APIClient
from benchmark.catalog import ModelCatalog
from benchmark.metrics import MetricsCollector, VRAMTracker
from benchmark.models import (
    APIResponse,
    BenchmarkRunConfig,
    GenerationParams,
    ModelEntry,
    Prompt,
    PromptCategory,
    PromptResult,
    ResponseMetrics,
    TestSet,
    VRAMSnapshot,
)
from benchmark.runner import BenchmarkRunner
from benchmark.storage import StorageManager
from benchmark.test_sets import TestSetManager


# --- Fixtures ---


@pytest.fixture
def tmp_data_path(tmp_path: Path) -> Path:
    """Geçici benchmark_data dizini."""
    data_dir = tmp_path / "benchmark_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def storage(tmp_data_path: Path) -> StorageManager:
    return StorageManager(tmp_data_path)


@pytest.fixture
def catalog(tmp_data_path: Path) -> ModelCatalog:
    cat = ModelCatalog(tmp_data_path)
    cat.register(
        ModelEntry(
            name="test-model",
            repo="org/test-model",
            format="FP16",
            backend="vllm",
            tags=["test"],
            api_endpoint="http://localhost:8000",
        )
    )
    return cat


@pytest.fixture
def test_set_manager(tmp_data_path: Path) -> TestSetManager:
    mgr = TestSetManager(tmp_data_path)
    ts = TestSet(
        name="test-set",
        description="Test seti",
        prompts=[
            Prompt(id="p1", category=PromptCategory.CODING, text="Hello"),
            Prompt(id="p2", category=PromptCategory.TRANSLATION, text="World"),
        ],
    )
    mgr.create(ts)
    return mgr


@pytest.fixture
def mock_api_client() -> AsyncMock:
    client = AsyncMock(spec=APIClient)
    client.chat_completion.return_value = APIResponse(
        status_code=200,
        response_text="Test response",
        usage={"prompt_tokens": 10, "completion_tokens": 20},
        error=None,
        elapsed_ms=500.0,
    )
    return client


@pytest.fixture
def metrics_collector() -> MetricsCollector:
    return MetricsCollector()


@pytest.fixture
def runner(
    catalog: ModelCatalog,
    test_set_manager: TestSetManager,
    mock_api_client: AsyncMock,
    metrics_collector: MetricsCollector,
    storage: StorageManager,
) -> BenchmarkRunner:
    return BenchmarkRunner(
        catalog=catalog,
        test_set_manager=test_set_manager,
        api_client=mock_api_client,
        metrics_collector=metrics_collector,
        storage=storage,
    )


# --- _generate_run_id Tests ---


class TestGenerateRunId:
    def test_format_matches_pattern(self, runner: BenchmarkRunner) -> None:
        """Çalıştırma ID'si YYYYMMDD_HHMMSS_xxxxxx formatında olmalı."""
        run_id = runner._generate_run_id()
        pattern = r"^\d{8}_\d{6}_[a-f0-9]{6}$"
        assert re.match(pattern, run_id), f"ID format hatası: {run_id}"

    def test_unique_ids(self, runner: BenchmarkRunner) -> None:
        """Ardışık çağrılar farklı ID'ler üretmeli."""
        ids = {runner._generate_run_id() for _ in range(10)}
        assert len(ids) == 10


# --- run() Tests ---


class TestRun:
    @pytest.mark.asyncio
    async def test_run_creates_run_dir_and_meta(
        self, runner: BenchmarkRunner, storage: StorageManager
    ) -> None:
        """run() çalıştırma dizini ve meta.json oluşturmalı."""
        config = BenchmarkRunConfig(
            test_set_name="test-set",
            model_names=["test-model"],
        )
        result = await runner.run(config)

        run_dir = storage.runs_dir / result.run_id
        assert run_dir.exists()
        meta = storage.read_json(run_dir / "meta.json")
        assert meta["run_id"] == result.run_id
        assert meta["config"]["test_set_name"] == "test-set"

    @pytest.mark.asyncio
    async def test_run_returns_all_results(
        self, runner: BenchmarkRunner
    ) -> None:
        """2 prompt × 1 model = 2 sonuç döndürmeli."""
        config = BenchmarkRunConfig(
            test_set_name="test-set",
            model_names=["test-model"],
        )
        result = await runner.run(config)

        assert len(result.results) == 2
        assert result.completed is True

    @pytest.mark.asyncio
    async def test_run_writes_results_to_jsonl(
        self, runner: BenchmarkRunner, storage: StorageManager
    ) -> None:
        """Her sonuç JSONL dosyasına yazılmalı."""
        config = BenchmarkRunConfig(
            test_set_name="test-set",
            model_names=["test-model"],
        )
        result = await runner.run(config)

        stored = storage.read_results(result.run_id)
        assert len(stored) == 2

    @pytest.mark.asyncio
    async def test_run_stores_category_in_result(
        self, runner: BenchmarkRunner, storage: StorageManager
    ) -> None:
        """Sonuç dict'inde prompt kategorisi saklanmalı."""
        config = BenchmarkRunConfig(
            test_set_name="test-set",
            model_names=["test-model"],
        )
        result = await runner.run(config)

        stored = storage.read_results(result.run_id)
        categories = {r["category"] for r in stored}
        assert "kodlama" in categories
        assert "çeviri" in categories

    @pytest.mark.asyncio
    async def test_run_with_repetition(
        self, runner: BenchmarkRunner
    ) -> None:
        """repetition_count=2 ile 2 prompt × 1 model × 2 tekrar = 4 sonuç."""
        config = BenchmarkRunConfig(
            test_set_name="test-set",
            model_names=["test-model"],
            params=GenerationParams(repetition_count=2),
        )
        result = await runner.run(config)
        assert len(result.results) == 4

    @pytest.mark.asyncio
    async def test_run_skips_unknown_model(
        self, runner: BenchmarkRunner
    ) -> None:
        """Bilinmeyen model adı atlanmalı, hata fırlatılmamalı."""
        config = BenchmarkRunConfig(
            test_set_name="test-set",
            model_names=["nonexistent-model"],
        )
        result = await runner.run(config)
        assert len(result.results) == 0
        assert result.completed is True


# --- _run_single Tests ---


class TestRunSingle:
    @pytest.mark.asyncio
    async def test_successful_single_run(
        self, runner: BenchmarkRunner
    ) -> None:
        """Başarılı API yanıtı PromptResult(success=True) döndürmeli."""
        model = runner.catalog.get("test-model")
        prompt = Prompt(id="p1", category=PromptCategory.CODING, text="Hi")
        params = GenerationParams()

        result = await runner._run_single(model, prompt, params, "test-run")

        assert result.success is True
        assert result.response_text == "Test response"
        assert result.metrics is not None
        assert result.error is None

    @pytest.mark.asyncio
    async def test_api_error_returns_failed_result(
        self, runner: BenchmarkRunner, mock_api_client: AsyncMock
    ) -> None:
        """API hatası durumunda success=False ve error alanı dolu olmalı."""
        mock_api_client.chat_completion.return_value = APIResponse(
            status_code=500,
            response_text=None,
            usage=None,
            error="Internal Server Error",
            elapsed_ms=100.0,
        )

        model = runner.catalog.get("test-model")
        prompt = Prompt(id="p1", category=PromptCategory.CODING, text="Hi")
        params = GenerationParams()

        result = await runner._run_single(model, prompt, params, "test-run")

        assert result.success is False
        assert result.error == "Internal Server Error"
        assert result.response_text is None

    @pytest.mark.asyncio
    async def test_exception_returns_failed_result(
        self, runner: BenchmarkRunner, mock_api_client: AsyncMock
    ) -> None:
        """Beklenmeyen exception durumunda success=False olmalı."""
        mock_api_client.chat_completion.side_effect = RuntimeError("Connection lost")

        model = runner.catalog.get("test-model")
        prompt = Prompt(id="p1", category=PromptCategory.CODING, text="Hi")
        params = GenerationParams()

        result = await runner._run_single(model, prompt, params, "test-run")

        assert result.success is False
        assert "Connection lost" in result.error

    @pytest.mark.asyncio
    async def test_vram_snapshot_included_when_tracker_available(
        self, runner: BenchmarkRunner
    ) -> None:
        """VRAMTracker varsa snapshot sonuca dahil edilmeli."""
        from datetime import datetime, timezone

        mock_tracker = MagicMock(spec=VRAMTracker)
        mock_tracker.snapshot.return_value = VRAMSnapshot(
            timestamp=datetime.now(timezone.utc),
            used_mb=8000.0,
            total_mb=16384.0,
            source="nvidia_smi",
        )
        runner.vram_tracker = mock_tracker

        model = runner.catalog.get("test-model")
        prompt = Prompt(id="p1", category=PromptCategory.CODING, text="Hi")
        params = GenerationParams()

        result = await runner._run_single(model, prompt, params, "test-run")

        assert result.success is True
        assert result.vram_snapshot is not None
        assert result.vram_snapshot.used_mb == 8000.0


# --- resume() Tests ---


class TestResume:
    @pytest.mark.asyncio
    async def test_resume_skips_completed_prompts(
        self,
        catalog: ModelCatalog,
        test_set_manager: TestSetManager,
        mock_api_client: AsyncMock,
        metrics_collector: MetricsCollector,
        storage: StorageManager,
    ) -> None:
        """Resume, tamamlanan prompt'ları atlayarak devam etmeli."""
        runner = BenchmarkRunner(
            catalog=catalog,
            test_set_manager=test_set_manager,
            api_client=mock_api_client,
            metrics_collector=metrics_collector,
            storage=storage,
        )

        # İlk çalıştırma — sadece 1 prompt tamamlanmış gibi simüle et
        run_id = runner._generate_run_id()
        storage.create_run_dir(run_id)
        config = BenchmarkRunConfig(
            test_set_name="test-set",
            model_names=["test-model"],
        )
        meta = {
            "run_id": run_id,
            "timestamp": "2025-01-01T00:00:00+00:00",
            "config": config.model_dump(mode="json"),
        }
        storage.write_json(storage.runs_dir / run_id / "meta.json", meta)

        # p1 tamamlanmış olarak kaydet
        storage.append_result(run_id, {
            "prompt_id": "p1",
            "model_name": "test-model",
            "response_text": "done",
            "success": True,
        })

        # Resume — p1 atlanmalı, sadece p2 çalıştırılmalı
        result = await runner.resume(run_id)

        # Toplam sonuç: 1 (mevcut) + 1 (yeni p2) = 2
        assert len(result.results) == 2
        assert result.completed is True

        # API sadece 1 kez çağrılmalı (p2 için)
        assert mock_api_client.chat_completion.call_count == 1


# --- Error Handling in run() ---


class TestRunErrorHandling:
    @pytest.mark.asyncio
    async def test_run_continues_on_api_error(
        self, runner: BenchmarkRunner, mock_api_client: AsyncMock
    ) -> None:
        """API hatası olsa bile çalıştırma devam etmeli."""
        call_count = 0

        async def alternating_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return APIResponse(
                    status_code=500,
                    response_text=None,
                    usage=None,
                    error="Server Error",
                    elapsed_ms=100.0,
                )
            return APIResponse(
                status_code=200,
                response_text="OK",
                usage={"prompt_tokens": 5, "completion_tokens": 10},
                error=None,
                elapsed_ms=200.0,
            )

        mock_api_client.chat_completion.side_effect = alternating_response

        config = BenchmarkRunConfig(
            test_set_name="test-set",
            model_names=["test-model"],
        )
        result = await runner.run(config)

        assert len(result.results) == 2
        failed = [r for r in result.results if not r.success]
        succeeded = [r for r in result.results if r.success]
        assert len(failed) == 1
        assert len(succeeded) == 1

    @pytest.mark.asyncio
    async def test_run_config_params_stored_in_meta(
        self, runner: BenchmarkRunner, storage: StorageManager
    ) -> None:
        """Kullanılan parametreler meta.json'da kaydedilmeli."""
        config = BenchmarkRunConfig(
            test_set_name="test-set",
            model_names=["test-model"],
            params=GenerationParams(temperature=0.5, max_tokens=512),
        )
        result = await runner.run(config)

        meta = storage.read_json(storage.runs_dir / result.run_id / "meta.json")
        assert meta["config"]["params"]["temperature"] == 0.5
        assert meta["config"]["params"]["max_tokens"] == 512
