"""Metrik Toplayıcı ve VRAM Tracker.

Performans metriklerinin ölçülmesi (TTFT, tokens/s, toplam süre)
ve GPU bellek kullanımının takibi.
"""

import subprocess
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

from benchmark.models import ResponseMetrics, VRAMSnapshot


class MetricsCollector:
    """Her API yanıtı için performans metriklerini toplar.

    Kullanım:
        collector = MetricsCollector()
        collector.start_timer()
        # ... ilk token geldiğinde:
        collector.record_first_token()
        # ... yanıt tamamlandığında:
        collector.stop_timer()
        metrics = collector.calculate(prompt_tokens=128, completion_tokens=256)
    """

    def __init__(self) -> None:
        self._start_time: Optional[float] = None
        self._first_token_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def start_timer(self) -> None:
        """Zamanlayıcıyı başlat."""
        self._start_time = time.perf_counter()
        self._first_token_time = None
        self._end_time = None

    def record_first_token(self) -> None:
        """İlk token zamanını kaydet (TTFT)."""
        self._first_token_time = time.perf_counter()

    def stop_timer(self) -> None:
        """Zamanlayıcıyı durdur."""
        self._end_time = time.perf_counter()

    def calculate(self, prompt_tokens: int, completion_tokens: int) -> ResponseMetrics:
        """Toplanan zamanlama verilerinden ResponseMetrics hesapla.

        Args:
            prompt_tokens: Prompt'taki token sayısı.
            completion_tokens: Üretilen token sayısı.

        Returns:
            Hesaplanmış performans metrikleri.

        Raises:
            RuntimeError: Zamanlayıcı başlatılmamış veya durdurulmamışsa.
        """
        if self._start_time is None or self._end_time is None:
            raise RuntimeError("Timer must be started and stopped before calculating metrics")

        total_time_ms = (self._end_time - self._start_time) * 1000

        ttft_ms: Optional[float] = None
        if self._first_token_time is not None:
            ttft_ms = (self._first_token_time - self._start_time) * 1000

        tokens_per_second: Optional[float] = None
        if completion_tokens > 0 and total_time_ms > 0:
            tokens_per_second = completion_tokens / (total_time_ms / 1000)

        return ResponseMetrics(
            ttft_ms=ttft_ms,
            total_time_ms=total_time_ms,
            tokens_per_second=tokens_per_second,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class VRAMTracker:
    """GPU bellek kullanımını izler.

    İki kaynaktan veri toplar:
    1. nvidia-smi komutu (Colab veya GPU'lu makinede)
    2. vLLM /metrics endpoint'i (Prometheus metrikleri)

    Her ikisi de yoksa sessizce None döndürür.
    """

    def __init__(self, metrics_endpoint: Optional[str] = None) -> None:
        self._metrics_endpoint = metrics_endpoint

    def is_available(self) -> bool:
        """nvidia-smi veya metrics endpoint erişilebilir mi kontrol et."""
        if self._metrics_endpoint is not None:
            return True
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def snapshot(self) -> Optional[VRAMSnapshot]:
        """Anlık VRAM kullanım bilgisi al.

        Önce nvidia-smi dener, sonra metrics endpoint'i dener.
        İkisi de yoksa None döndürür.
        """
        # nvidia-smi'yi dene
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return self.parse_nvidia_smi(result.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # vLLM metrics endpoint'ini dene
        if self._metrics_endpoint is not None:
            return self.fetch_vllm_metrics()

        return None

    def parse_nvidia_smi(self, output: str) -> VRAMSnapshot:
        """nvidia-smi CSV çıktısını parse et.

        Beklenen format (--query-gpu=memory.used,memory.total --format=csv,noheader,nounits):
            8192, 16384

        Birden fazla GPU varsa ilk satırı kullanır.

        Args:
            output: nvidia-smi komut çıktısı.

        Returns:
            VRAMSnapshot nesnesi.

        Raises:
            ValueError: Çıktı parse edilemezse.
        """
        line = output.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            raise ValueError(f"Invalid nvidia-smi output: {output!r}")
        try:
            used_mb = float(parts[0])
            total_mb = float(parts[1])
        except ValueError:
            raise ValueError(f"Cannot parse nvidia-smi values: {output!r}")

        return VRAMSnapshot(
            timestamp=datetime.now(timezone.utc),
            used_mb=used_mb,
            total_mb=total_mb,
            source="nvidia_smi",
        )

    def fetch_vllm_metrics(self) -> Optional[VRAMSnapshot]:
        """vLLM /metrics endpoint'inden VRAM bilgisi oku.

        Prometheus metrikleri arasında gpu_memory_usage_bytes veya
        benzer metrikleri arar.

        Returns:
            VRAMSnapshot veya erişilemezse None.
        """
        if self._metrics_endpoint is None:
            return None

        try:
            response = httpx.get(self._metrics_endpoint, timeout=5.0)
            if response.status_code != 200:
                return None

            text = response.text
            used_bytes: Optional[float] = None
            total_bytes: Optional[float] = None

            for raw_line in text.split("\n"):
                line = raw_line.strip()
                if line.startswith("#"):
                    continue

                if "gpu_memory_usage_bytes" in line:
                    used_bytes = self._parse_prometheus_value(line)
                elif "gpu_memory_total_bytes" in line:
                    total_bytes = self._parse_prometheus_value(line)

            if used_bytes is not None:
                used_mb = used_bytes / (1024 * 1024)
                total_mb = total_bytes / (1024 * 1024) if total_bytes is not None else 0.0
                return VRAMSnapshot(
                    timestamp=datetime.now(timezone.utc),
                    used_mb=used_mb,
                    total_mb=total_mb,
                    source="vllm_metrics",
                )

        except (httpx.RequestError, httpx.TimeoutException):
            pass

        return None

    @staticmethod
    def _parse_prometheus_value(line: str) -> Optional[float]:
        """Prometheus metrik satırından sayısal değeri çıkar.

        Örnek satır: vllm:gpu_memory_usage_bytes{gpu="0"} 1.234e+10
        """
        parts = line.split()
        if len(parts) >= 2:
            try:
                return float(parts[-1])
            except ValueError:
                pass
        return None
