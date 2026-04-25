"""Depolama yöneticisi — dosya sistemi I/O işlemlerinin soyutlanması."""

import csv
import json
from pathlib import Path


class StorageManager:
    """Benchmark verilerinin dosya sisteminde saklanmasını yöneten sınıf.

    Tüm dosya I/O işlemlerini soyutlar: JSON okuma/yazma, JSONL append/read,
    CSV dışa aktarma ve çalıştırma dizini yönetimi.
    """

    def __init__(self, base_path: Path) -> None:
        """Depolama yöneticisini başlat.

        Args:
            base_path: Benchmark verilerinin kök dizini (benchmark_data/).
        """
        self.base_path = Path(base_path)
        self.runs_dir = self.base_path / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, run_id: str) -> Path:
        """Yeni bir çalıştırma dizini oluştur.

        Args:
            run_id: Çalıştırma kimliği.

        Returns:
            Oluşturulan dizinin yolu.
        """
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def write_json(self, path: Path, data: dict) -> None:
        """Sözlüğü JSON dosyasına yaz.

        Args:
            path: Hedef dosya yolu.
            data: Yazılacak veri sözlüğü.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def read_json(self, path: Path) -> dict:
        """JSON dosyasını oku ve sözlük olarak döndür.

        Args:
            path: Okunacak dosya yolu.

        Returns:
            Dosyadaki JSON verisi.

        Raises:
            FileNotFoundError: Dosya bulunamazsa.
            json.JSONDecodeError: Geçersiz JSON formatı.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_csv(self, path: Path, rows: list[dict], fieldnames: list[str]) -> None:
        """CSV dosyasına yaz (UTF-8 BOM ile Türkçe karakter desteği).

        Args:
            path: Hedef dosya yolu.
            rows: Yazılacak satırlar (her biri bir sözlük).
            fieldnames: CSV sütun başlıkları.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def append_result(self, run_id: str, result: dict) -> None:
        """Sonucu JSONL formatında çalıştırma dosyasına ekle.

        Args:
            run_id: Çalıştırma kimliği.
            result: Eklenecek sonuç sözlüğü.
        """
        results_path = self.runs_dir / run_id / "results.jsonl"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def read_results(self, run_id: str) -> list[dict]:
        """Çalıştırma sonuçlarını JSONL dosyasından oku.

        Args:
            run_id: Çalıştırma kimliği.

        Returns:
            Sonuç sözlüklerinin listesi.
        """
        results_path = self.runs_dir / run_id / "results.jsonl"
        if not results_path.exists():
            return []
        results = []
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results

    def list_runs(self) -> list[str]:
        """Mevcut çalıştırma dizinlerini listele.

        Returns:
            Çalıştırma kimliklerinin listesi.
        """
        if not self.runs_dir.exists():
            return []
        return [d.name for d in self.runs_dir.iterdir() if d.is_dir()]

    def get_completed_prompt_ids(self, run_id: str, model_name: str) -> set[str]:
        """Resume desteği: belirli bir model için tamamlanan prompt ID'lerini döndür.

        Args:
            run_id: Çalıştırma kimliği.
            model_name: Filtrelenecek model adı.

        Returns:
            Tamamlanan prompt ID'lerinin kümesi.
        """
        results = self.read_results(run_id)
        return {
            r["prompt_id"]
            for r in results
            if r.get("model_name") == model_name
        }
