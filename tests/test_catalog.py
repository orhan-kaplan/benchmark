"""ModelCatalog birim testleri."""

import json
from pathlib import Path

import pytest

from benchmark.catalog import ModelCatalog
from benchmark.models import Backend, ModelEntry


# --- Helpers ---


def _make_entry(
    name: str = "test-model",
    backend: Backend = Backend.VLLM,
    tags: list[str] | None = None,
    endpoint: str = "http://localhost:8000",
) -> ModelEntry:
    return ModelEntry(
        name=name,
        repo="org/test-model",
        format="FP16",
        backend=backend,
        tags=tags or [],
        api_endpoint=endpoint,
    )


# --- Init / persistence ---


class TestInit:
    def test_creates_empty_file_when_missing(self, tmp_path: Path) -> None:
        ModelCatalog(tmp_path)
        models_file = tmp_path / "models.json"
        assert models_file.exists()
        data = json.loads(models_file.read_text("utf-8"))
        assert data == {"models": []}

    def test_loads_existing_file(self, tmp_path: Path) -> None:
        entry = _make_entry()
        models_file = tmp_path / "models.json"
        models_file.write_text(
            json.dumps({"models": [entry.model_dump(mode="json")]}),
            encoding="utf-8",
        )
        catalog = ModelCatalog(tmp_path)
        assert len(catalog.list_all()) == 1
        assert catalog.list_all()[0].name == "test-model"


# --- register ---


class TestRegister:
    def test_register_adds_model(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_make_entry())
        assert len(catalog.list_all()) == 1

    def test_register_persists_to_disk(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_make_entry())
        # Reload from disk
        catalog2 = ModelCatalog(tmp_path)
        assert len(catalog2.list_all()) == 1

    def test_register_duplicate_raises(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_make_entry("dup"))
        with pytest.raises(ValueError, match="zaten kayıtlı"):
            catalog.register(_make_entry("dup"))


# --- get ---


class TestGet:
    def test_get_existing(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_make_entry("alpha"))
        result = catalog.get("alpha")
        assert result is not None
        assert result.name == "alpha"

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        assert catalog.get("nonexistent") is None


# --- update ---


class TestUpdate:
    def test_update_returns_updated_entry(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_make_entry("upd"))
        updated = catalog.update("upd", {"format": "GPTQ"})
        assert updated.format == "GPTQ"

    def test_update_persists(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_make_entry("upd"))
        catalog.update("upd", {"tags": ["translation"]})
        catalog2 = ModelCatalog(tmp_path)
        assert catalog2.get("upd").tags == ["translation"]

    def test_update_missing_raises(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        with pytest.raises(KeyError):
            catalog.update("ghost", {"format": "GGUF"})


# --- remove ---


class TestRemove:
    def test_remove_deletes_model(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_make_entry("rm"))
        catalog.remove("rm")
        assert catalog.get("rm") is None

    def test_remove_persists(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_make_entry("rm"))
        catalog.remove("rm")
        catalog2 = ModelCatalog(tmp_path)
        assert len(catalog2.list_all()) == 0

    def test_remove_missing_raises(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        with pytest.raises(KeyError):
            catalog.remove("ghost")


# --- filter_by ---


class TestFilterBy:
    def test_filter_by_backend(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_make_entry("v1", backend=Backend.VLLM))
        catalog.register(_make_entry("o1", backend=Backend.OLLAMA))
        result = catalog.filter_by(backend="vllm")
        assert len(result) == 1
        assert result[0].name == "v1"

    def test_filter_by_tags_all_match(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_make_entry("t1", tags=["translation", "reasoning"]))
        catalog.register(_make_entry("t2", tags=["translation"]))
        result = catalog.filter_by(tags=["translation", "reasoning"])
        assert len(result) == 1
        assert result[0].name == "t1"

    def test_filter_by_backend_and_tags(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(
            _make_entry("combo", backend=Backend.VLLM, tags=["coding"])
        )
        catalog.register(
            _make_entry("other", backend=Backend.OLLAMA, tags=["coding"])
        )
        result = catalog.filter_by(backend="vllm", tags=["coding"])
        assert len(result) == 1
        assert result[0].name == "combo"

    def test_filter_no_criteria_returns_all(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_make_entry("a"))
        catalog.register(_make_entry("b"))
        assert len(catalog.filter_by()) == 2


# --- list_all ---


class TestListAll:
    def test_empty_catalog(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        assert catalog.list_all() == []

    def test_returns_copy(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_make_entry())
        lst = catalog.list_all()
        lst.clear()
        assert len(catalog.list_all()) == 1
