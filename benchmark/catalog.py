"""Model Kataloğu — model meta verilerinin CRUD işlemleri ve kalıcı JSON depolama."""

import json
from pathlib import Path
from typing import Optional

from benchmark.models import ModelEntry


class ModelCatalog:
    """Model meta verilerinin merkezi kayıt defteri.

    Modelleri JSON dosyasında kalıcı olarak saklar ve bellek içi
    filtreleme/arama işlemleri sunar. Tüm değişiklikler anında
    diske yazılır (write-through).
    """

    def __init__(self, data_path: Path) -> None:
        """Kataloğu başlat.

        Args:
            data_path: ``benchmark_data/`` kök dizini. Katalog dosyası
                       ``data_path / "models.json"`` olarak saklanır.
        """
        self._file_path = Path(data_path) / "models.json"
        self._models: list[ModelEntry] = []
        self._load()

    # --- Public API ---

    def register(self, model: ModelEntry) -> None:
        """Yeni bir model kaydet.

        Args:
            model: Kaydedilecek model girişi.

        Raises:
            ValueError: Aynı isimde bir model zaten kayıtlıysa.
        """
        if any(m.name == model.name for m in self._models):
            raise ValueError(
                f"'{model.name}' adında bir model zaten kayıtlı."
            )
        self._models.append(model)
        self._save()

    def update(self, model_name: str, updates: dict) -> ModelEntry:
        """Mevcut bir model kaydını güncelle.

        Args:
            model_name: Güncellenecek modelin adı.
            updates: Uygulanacak alan güncellemeleri sözlüğü.

        Returns:
            Güncellenmiş ``ModelEntry`` nesnesi.

        Raises:
            KeyError: Belirtilen isimde model bulunamazsa.
        """
        for i, m in enumerate(self._models):
            if m.name == model_name:
                data = m.model_dump()
                data.update(updates)
                updated = ModelEntry(**data)
                self._models[i] = updated
                self._save()
                return updated
        raise KeyError(f"'{model_name}' adında bir model bulunamadı.")

    def get(self, model_name: str) -> Optional[ModelEntry]:
        """Tek bir modeli ada göre getir.

        Args:
            model_name: Aranan model adı.

        Returns:
            Bulunan ``ModelEntry`` veya ``None``.
        """
        for m in self._models:
            if m.name == model_name:
                return m
        return None

    def list_all(self) -> list[ModelEntry]:
        """Tüm kayıtlı modelleri döndür."""
        return list(self._models)

    def filter_by(
        self,
        backend: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[ModelEntry]:
        """Modelleri backend ve/veya etiketlere göre filtrele.

        Args:
            backend: Filtrelenecek backend türü (ör. ``"vllm"``).
            tags: Filtrelenecek etiket listesi. Model, belirtilen
                  etiketlerin **tamamına** sahip olmalıdır.

        Returns:
            Filtre kriterlerini karşılayan modellerin listesi.
        """
        result = self._models

        if backend is not None:
            result = [m for m in result if m.backend.value == backend]

        if tags is not None:
            result = [
                m for m in result if all(t in m.tags for t in tags)
            ]

        return list(result)

    def remove(self, model_name: str) -> None:
        """Model kaydını sil.

        Args:
            model_name: Silinecek modelin adı.

        Raises:
            KeyError: Belirtilen isimde model bulunamazsa.
        """
        for i, m in enumerate(self._models):
            if m.name == model_name:
                del self._models[i]
                self._save()
                return
        raise KeyError(f"'{model_name}' adında bir model bulunamadı.")

    # --- Persistence ---

    def _save(self) -> None:
        """Katalog verisini ``models.json`` dosyasına yaz."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "models": [m.model_dump(mode="json") for m in self._models]
        }
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        """``models.json`` dosyasından katalog verisini yükle.

        Dosya yoksa boş bir katalog dosyası oluşturur.
        """
        if not self._file_path.exists():
            self._models = []
            self._save()
            return

        with open(self._file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._models = [
            ModelEntry(**entry) for entry in data.get("models", [])
        ]
