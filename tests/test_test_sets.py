"""TestSetManager birim testleri."""

import pytest

from benchmark.models import Difficulty, Prompt, PromptCategory, TestSet
from benchmark.test_sets import TestSetManager


@pytest.fixture
def manager(tmp_path):
    """Geçici dizinde TestSetManager oluştur."""
    return TestSetManager(data_path=tmp_path)


@pytest.fixture
def sample_prompt():
    return Prompt(
        id="p-001",
        category=PromptCategory.TRANSLATION,
        text="Translate this sentence.",
        difficulty=Difficulty.EASY,
    )


@pytest.fixture
def sample_test_set(sample_prompt):
    return TestSet(
        name="translation",
        description="Translation test set",
        prompts=[sample_prompt],
    )


class TestCreate:
    def test_create_and_load(self, manager, sample_test_set):
        manager.create(sample_test_set)
        loaded = manager.load("translation")
        assert loaded.name == sample_test_set.name
        assert len(loaded.prompts) == 1
        assert loaded.prompts[0].id == "p-001"

    def test_create_empty_set(self, manager):
        ts = TestSet(name="empty")
        manager.create(ts)
        loaded = manager.load("empty")
        assert loaded.prompts == []


class TestLoad:
    def test_load_not_found(self, manager):
        with pytest.raises(FileNotFoundError):
            manager.load("nonexistent")


class TestAddPrompt:
    def test_add_prompt_increases_count(self, manager, sample_test_set):
        manager.create(sample_test_set)
        new_prompt = Prompt(
            id="p-002",
            category=PromptCategory.CODING,
            text="Write a function.",
            difficulty=Difficulty.MEDIUM,
        )
        manager.add_prompt("translation", new_prompt)
        loaded = manager.load("translation")
        assert len(loaded.prompts) == 2
        assert loaded.prompts[1].id == "p-002"

    def test_add_prompt_duplicate_id_raises(self, manager, sample_test_set, sample_prompt):
        manager.create(sample_test_set)
        with pytest.raises(ValueError, match="p-001"):
            manager.add_prompt("translation", sample_prompt)

    def test_add_prompt_to_nonexistent_set(self, manager, sample_prompt):
        with pytest.raises(FileNotFoundError):
            manager.add_prompt("missing", sample_prompt)


class TestListSets:
    def test_list_empty(self, manager):
        assert manager.list_sets() == []

    def test_list_after_create(self, manager, sample_test_set):
        manager.create(sample_test_set)
        names = manager.list_sets()
        assert "translation" in names

    def test_list_multiple(self, manager):
        manager.create(TestSet(name="a"))
        manager.create(TestSet(name="b"))
        names = sorted(manager.list_sets())
        assert names == ["a", "b"]


class TestFilterPrompts:
    def test_filter_by_category(self, manager):
        ts = TestSet(
            name="mixed",
            prompts=[
                Prompt(id="1", category=PromptCategory.TRANSLATION, text="t1"),
                Prompt(id="2", category=PromptCategory.CODING, text="t2"),
                Prompt(id="3", category=PromptCategory.TRANSLATION, text="t3"),
            ],
        )
        manager.create(ts)
        result = manager.filter_prompts("mixed", category="translation")
        assert len(result) == 2
        assert all(p.category == PromptCategory.TRANSLATION for p in result)

    def test_filter_by_difficulty(self, manager):
        ts = TestSet(
            name="diff",
            prompts=[
                Prompt(id="1", category=PromptCategory.GENERAL, text="t1", difficulty=Difficulty.EASY),
                Prompt(id="2", category=PromptCategory.GENERAL, text="t2", difficulty=Difficulty.HARD),
            ],
        )
        manager.create(ts)
        result = manager.filter_prompts("diff", difficulty="easy")
        assert len(result) == 1
        assert result[0].difficulty == Difficulty.EASY

    def test_filter_by_both(self, manager):
        ts = TestSet(
            name="both",
            prompts=[
                Prompt(id="1", category=PromptCategory.CODING, text="t1", difficulty=Difficulty.EASY),
                Prompt(id="2", category=PromptCategory.CODING, text="t2", difficulty=Difficulty.HARD),
                Prompt(id="3", category=PromptCategory.GENERAL, text="t3", difficulty=Difficulty.EASY),
            ],
        )
        manager.create(ts)
        result = manager.filter_prompts("both", category="coding", difficulty="easy")
        assert len(result) == 1
        assert result[0].id == "1"

    def test_filter_no_criteria_returns_all(self, manager, sample_test_set):
        manager.create(sample_test_set)
        result = manager.filter_prompts("translation")
        assert len(result) == len(sample_test_set.prompts)


class TestSerializeDeserialize:
    def test_roundtrip(self, manager, sample_test_set):
        json_str = manager.serialize(sample_test_set)
        restored = manager.deserialize(json_str)
        assert restored == sample_test_set

    def test_serialize_is_pretty(self, manager, sample_test_set):
        json_str = manager.serialize(sample_test_set)
        assert "\n" in json_str  # pretty-printed
