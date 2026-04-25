"""StorageManager birim testleri."""

import json
from pathlib import Path

import pytest

from benchmark.storage import StorageManager


@pytest.fixture
def storage(tmp_path: Path) -> StorageManager:
    """Geçici dizinde StorageManager oluştur."""
    return StorageManager(tmp_path)


class TestInit:
    def test_creates_runs_dir(self, tmp_path: Path) -> None:
        sm = StorageManager(tmp_path)
        assert (tmp_path / "runs").is_dir()
        assert sm.base_path == tmp_path
        assert sm.runs_dir == tmp_path / "runs"

    def test_idempotent_init(self, tmp_path: Path) -> None:
        StorageManager(tmp_path)
        StorageManager(tmp_path)
        assert (tmp_path / "runs").is_dir()


class TestCreateRunDir:
    def test_creates_directory(self, storage: StorageManager) -> None:
        run_dir = storage.create_run_dir("run_001")
        assert run_dir.is_dir()
        assert run_dir.name == "run_001"

    def test_returns_path(self, storage: StorageManager) -> None:
        run_dir = storage.create_run_dir("run_002")
        assert run_dir == storage.runs_dir / "run_002"

    def test_idempotent(self, storage: StorageManager) -> None:
        storage.create_run_dir("run_003")
        run_dir = storage.create_run_dir("run_003")
        assert run_dir.is_dir()


class TestWriteReadJson:
    def test_roundtrip(self, storage: StorageManager) -> None:
        data = {"key": "value", "number": 42, "nested": {"a": 1}}
        path = storage.base_path / "test.json"
        storage.write_json(path, data)
        result = storage.read_json(path)
        assert result == data

    def test_turkish_characters(self, storage: StorageManager) -> None:
        data = {"mesaj": "Türkçe karakterler: ç, ğ, ı, ö, ş, ü"}
        path = storage.base_path / "turkish.json"
        storage.write_json(path, data)
        result = storage.read_json(path)
        assert result == data

    def test_pretty_print(self, storage: StorageManager) -> None:
        data = {"a": 1}
        path = storage.base_path / "pretty.json"
        storage.write_json(path, data)
        content = path.read_text(encoding="utf-8")
        assert "  " in content  # indent=2

    def test_creates_parent_dirs(self, storage: StorageManager) -> None:
        path = storage.base_path / "sub" / "dir" / "file.json"
        storage.write_json(path, {"ok": True})
        assert storage.read_json(path) == {"ok": True}

    def test_read_nonexistent_raises(self, storage: StorageManager) -> None:
        with pytest.raises(FileNotFoundError):
            storage.read_json(storage.base_path / "nope.json")

    def test_read_invalid_json_raises(self, storage: StorageManager) -> None:
        path = storage.base_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            storage.read_json(path)


class TestAppendReadResults:
    def test_append_and_read(self, storage: StorageManager) -> None:
        storage.create_run_dir("run_a")
        storage.append_result("run_a", {"prompt_id": "p1", "model_name": "m1"})
        storage.append_result("run_a", {"prompt_id": "p2", "model_name": "m1"})
        results = storage.read_results("run_a")
        assert len(results) == 2
        assert results[0]["prompt_id"] == "p1"
        assert results[1]["prompt_id"] == "p2"

    def test_read_empty_run(self, storage: StorageManager) -> None:
        storage.create_run_dir("run_empty")
        results = storage.read_results("run_empty")
        assert results == []

    def test_read_nonexistent_run(self, storage: StorageManager) -> None:
        results = storage.read_results("no_such_run")
        assert results == []

    def test_jsonl_format(self, storage: StorageManager) -> None:
        storage.create_run_dir("run_fmt")
        storage.append_result("run_fmt", {"a": 1})
        storage.append_result("run_fmt", {"b": 2})
        path = storage.runs_dir / "run_fmt" / "results.jsonl"
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}


class TestWriteCsv:
    def test_basic_csv(self, storage: StorageManager) -> None:
        path = storage.base_path / "out.csv"
        rows = [{"name": "a", "score": "1"}, {"name": "b", "score": "2"}]
        storage.write_csv(path, rows, ["name", "score"])
        content = path.read_bytes()
        # UTF-8 BOM check
        assert content.startswith(b"\xef\xbb\xbf")
        text = content.decode("utf-8-sig")
        lines = text.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows

    def test_turkish_csv(self, storage: StorageManager) -> None:
        path = storage.base_path / "tr.csv"
        rows = [{"ad": "Öğrenci", "puan": "8"}]
        storage.write_csv(path, rows, ["ad", "puan"])
        text = path.read_bytes().decode("utf-8-sig")
        assert "Öğrenci" in text


class TestListRuns:
    def test_empty(self, storage: StorageManager) -> None:
        assert storage.list_runs() == []

    def test_lists_directories(self, storage: StorageManager) -> None:
        storage.create_run_dir("run_x")
        storage.create_run_dir("run_y")
        runs = storage.list_runs()
        assert set(runs) == {"run_x", "run_y"}

    def test_ignores_files(self, storage: StorageManager) -> None:
        storage.create_run_dir("run_z")
        (storage.runs_dir / "not_a_dir.txt").write_text("hi")
        runs = storage.list_runs()
        assert runs == ["run_z"]


class TestGetCompletedPromptIds:
    def test_filters_by_model(self, storage: StorageManager) -> None:
        storage.create_run_dir("run_r")
        storage.append_result("run_r", {"prompt_id": "p1", "model_name": "m1"})
        storage.append_result("run_r", {"prompt_id": "p2", "model_name": "m2"})
        storage.append_result("run_r", {"prompt_id": "p3", "model_name": "m1"})
        ids = storage.get_completed_prompt_ids("run_r", "m1")
        assert ids == {"p1", "p3"}

    def test_empty_results(self, storage: StorageManager) -> None:
        storage.create_run_dir("run_e")
        ids = storage.get_completed_prompt_ids("run_e", "m1")
        assert ids == set()

    def test_no_match(self, storage: StorageManager) -> None:
        storage.create_run_dir("run_n")
        storage.append_result("run_n", {"prompt_id": "p1", "model_name": "m1"})
        ids = storage.get_completed_prompt_ids("run_n", "m_other")
        assert ids == set()
