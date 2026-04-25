"""Benchmark Runner — test setlerini modellere gönderen ana orkestratör.

Prompt × model çiftlerini sıralı olarak çalıştırır, metrikleri toplar,
sonuçları anında diske yazar ve hata durumunda çalıştırmaya devam eder.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from benchmark.api_client import APIClient
from benchmark.catalog import ModelCatalog
from benchmark.metrics import MetricsCollector, VRAMTracker
from benchmark.models import (
    BenchmarkRun,
    BenchmarkRunConfig,
    GenerationParams,
    ModelEntry,
    Prompt,
    PromptResult,
)
from benchmark.storage import StorageManager
from benchmark.test_sets import TestSetManager

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Test setlerini modellere gönderen, metrikleri toplayan ve sonuçları diske yazan ana orkestratör.

    Her prompt × model çifti sıralı olarak çalıştırılır (adil karşılaştırma).
    Her sonuç anında JSONL formatında diske yazılır (crash koruması).
    """

    def __init__(
        self,
        catalog: ModelCatalog,
        test_set_manager: TestSetManager,
        api_client: APIClient,
        metrics_collector: MetricsCollector,
        storage: StorageManager,
        vram_tracker: Optional[VRAMTracker] = None,
    ) -> None:
        """Bağımlılık enjeksiyonu ile runner'ı başlat."""
        self.catalog = catalog
        self.test_set_manager = test_set_manager
        self.api_client = api_client
        self.metrics_collector = metrics_collector
        self.storage = storage
        self.vram_tracker = vram_tracker

    def _generate_run_id(self) -> str:
        """Benzersiz çalıştırma kimliği üret.

        Format: ``{YYYYMMDD}_{HHMMSS}_{uuid4_short}``
        Örnek: ``20250101_143022_a1b2c3``
        """
        now = datetime.now(timezone.utc)
        short_uuid = uuid.uuid4().hex[:6]
        return f"{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}_{short_uuid}"

    async def run(self, config: BenchmarkRunConfig) -> BenchmarkRun:
        """Ana benchmark orkestrasyon döngüsü.

        Args:
            config: Çalıştırma yapılandırması (test seti adı, model listesi, parametreler).

        Returns:
            Tüm sonuçları içeren ``BenchmarkRun`` nesnesi.
        """
        run_id = self._generate_run_id()
        timestamp = datetime.now(timezone.utc)

        # Çalıştırma dizini oluştur ve meta.json yaz
        self.storage.create_run_dir(run_id)
        run_dir = self.storage.runs_dir / run_id
        meta = {
            "run_id": run_id,
            "timestamp": timestamp.isoformat(),
            "config": config.model_dump(mode="json"),
        }
        self.storage.write_json(run_dir / "meta.json", meta)

        # Modelleri ve prompt'ları yükle
        models: list[ModelEntry] = []
        for name in config.model_names:
            model = self.catalog.get(name)
            if model is None:
                logger.warning("Model bulunamadı, atlanıyor: %s", name)
                continue
            models.append(model)

        test_set = self.test_set_manager.load(config.test_set_name)
        prompts = test_set.prompts

        total = len(prompts) * len(models) * config.params.repetition_count
        completed = 0
        results: list[PromptResult] = []

        for prompt in prompts:
            for model in models:
                for _rep in range(config.params.repetition_count):
                    result = await self._run_single(model, prompt, config.params, run_id)
                    results.append(result)

                    # Sonucu anında JSONL'e yaz (kategori bilgisi dahil)
                    result_dict = result.model_dump(mode="json")
                    result_dict["category"] = prompt.category.value
                    self.storage.append_result(run_id, result_dict)

                    completed += 1
                    logger.info("İlerleme: %d/%d tamamlandı", completed, total)

        benchmark_run = BenchmarkRun(
            run_id=run_id,
            timestamp=timestamp,
            config=config,
            results=results,
            completed=True,
        )
        return benchmark_run

    async def resume(self, run_id: str) -> BenchmarkRun:
        """Yarıda kalan çalıştırmayı kaldığı yerden devam ettir.

        Args:
            run_id: Devam ettirilecek çalıştırma kimliği.

        Returns:
            Tamamlanan ``BenchmarkRun`` nesnesi.
        """
        run_dir = self.storage.runs_dir / run_id
        meta = self.storage.read_json(run_dir / "meta.json")

        config = BenchmarkRunConfig(**meta["config"])
        timestamp = datetime.fromisoformat(meta["timestamp"])

        # Modelleri ve prompt'ları yükle
        models: list[ModelEntry] = []
        for name in config.model_names:
            model = self.catalog.get(name)
            if model is None:
                logger.warning("Model bulunamadı, atlanıyor: %s", name)
                continue
            models.append(model)

        test_set = self.test_set_manager.load(config.test_set_name)
        prompts = test_set.prompts

        # Mevcut sonuçları oku
        existing_results = self.storage.read_results(run_id)
        results: list[PromptResult] = [
            PromptResult(**r) for r in existing_results
        ]

        total = len(prompts) * len(models) * config.params.repetition_count
        completed = len(existing_results)

        for prompt in prompts:
            for model in models:
                # Tamamlanan prompt ID'lerini bu model için al
                completed_ids = self.storage.get_completed_prompt_ids(run_id, model.name)
                if prompt.id in completed_ids:
                    continue

                for _rep in range(config.params.repetition_count):
                    result = await self._run_single(model, prompt, config.params, run_id)
                    results.append(result)

                    result_dict = result.model_dump(mode="json")
                    result_dict["category"] = prompt.category.value
                    self.storage.append_result(run_id, result_dict)

                    completed += 1
                    logger.info("İlerleme (resume): %d/%d tamamlandı", completed, total)

        benchmark_run = BenchmarkRun(
            run_id=run_id,
            timestamp=timestamp,
            config=config,
            results=results,
            completed=True,
        )
        return benchmark_run

    async def _run_single(
        self,
        model: ModelEntry,
        prompt: Prompt,
        params: GenerationParams,
        run_id: str,
    ) -> PromptResult:
        """Tek bir prompt-model çifti için API çağrısı yap ve sonuç döndür.

        Args:
            model: Hedef model bilgisi.
            prompt: Gönderilecek prompt.
            params: Üretim parametreleri.
            run_id: Çalıştırma kimliği.

        Returns:
            ``PromptResult`` — başarılı veya hatalı sonuç.
        """
        try:
            # Zamanlayıcıyı başlat
            self.metrics_collector.start_timer()

            # API çağrısı
            messages = [{"role": "user", "content": prompt.text}]
            api_response = await self.api_client.chat_completion(
                endpoint=model.api_endpoint,
                model=model.repo,
                messages=messages,
                params=params,
            )

            # Zamanlayıcıyı durdur
            self.metrics_collector.stop_timer()

            # API hatası kontrolü
            if api_response.error is not None:
                logger.warning(
                    "API hatası [%s × %s]: %s",
                    model.name,
                    prompt.id,
                    api_response.error,
                )
                return PromptResult(
                    prompt_id=prompt.id,
                    model_name=model.name,
                    error=api_response.error,
                    success=False,
                )

            # Metrikleri hesapla
            prompt_tokens = 0
            completion_tokens = 0
            if api_response.usage:
                prompt_tokens = api_response.usage.get("prompt_tokens", 0)
                completion_tokens = api_response.usage.get("completion_tokens", 0)

            metrics = self.metrics_collector.calculate(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            # Override TTFT from streaming API response (more accurate)
            if api_response.ttft_ms is not None:
                metrics.ttft_ms = api_response.ttft_ms

            # VRAM snapshot (opsiyonel)
            vram_snapshot = None
            if self.vram_tracker is not None:
                vram_snapshot = self.vram_tracker.snapshot()

            return PromptResult(
                prompt_id=prompt.id,
                model_name=model.name,
                response_text=api_response.response_text,
                metrics=metrics,
                vram_snapshot=vram_snapshot,
                success=True,
            )

        except Exception as exc:
            logger.error(
                "Beklenmeyen hata [%s × %s]: %s",
                model.name,
                prompt.id,
                exc,
            )
            return PromptResult(
                prompt_id=prompt.id,
                model_name=model.name,
                error=str(exc),
                success=False,
            )
