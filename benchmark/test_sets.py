"""Test Seti Yöneticisi — test setlerinin oluşturulması, doğrulanması ve serileştirilmesi."""

from pathlib import Path
from typing import Optional

from benchmark.models import Prompt, TestSet


class TestSetManager:
    """Test setlerinin CRUD işlemleri ve JSON kalıcılığı.

    Her test seti ayrı bir JSON dosyasında saklanır
    (``test_sets/{name}.json``). Prompt ID benzersizliği
    test seti kapsamında kontrol edilir.
    """

    def __init__(self, data_path: Path) -> None:
        """Test seti yöneticisini başlat.

        Args:
            data_path: ``benchmark_data/`` kök dizini. Test setleri
                       ``data_path / "test_sets/"`` altında saklanır.
        """
        self._dir = Path(data_path) / "test_sets"
        self._dir.mkdir(parents=True, exist_ok=True)

    def create(self, test_set: TestSet) -> None:
        """Yeni bir test setini JSON dosyasına kaydet.

        Args:
            test_set: Kaydedilecek test seti.
        """
        path = self._dir / f"{test_set.name}.json"
        path.write_text(test_set.model_dump_json(indent=2), encoding="utf-8")

    def load(self, name: str) -> TestSet:
        """JSON dosyasından test setini yükle.

        Args:
            name: Test seti adı (dosya adı uzantısız).

        Returns:
            Yüklenen ``TestSet`` nesnesi.

        Raises:
            FileNotFoundError: Belirtilen isimde test seti bulunamazsa.
        """
        path = self._dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"'{name}' adında bir test seti bulunamadı: {path}"
            )
        return TestSet.model_validate_json(path.read_text(encoding="utf-8"))

    def add_prompt(self, set_name: str, prompt: Prompt) -> None:
        """Mevcut bir test setine prompt ekle.

        Args:
            set_name: Hedef test seti adı.
            prompt: Eklenecek prompt.

        Raises:
            FileNotFoundError: Test seti bulunamazsa.
            ValueError: Aynı ID'ye sahip bir prompt zaten varsa.
        """
        test_set = self.load(set_name)
        if any(p.id == prompt.id for p in test_set.prompts):
            raise ValueError(
                f"'{prompt.id}' ID'sine sahip bir prompt zaten mevcut."
            )
        test_set.prompts.append(prompt)
        self.create(test_set)

    def list_sets(self) -> list[str]:
        """Mevcut test seti isimlerini döndür.

        Returns:
            Test seti adlarının listesi (uzantısız dosya adları).
        """
        return [p.stem for p in self._dir.glob("*.json")]

    def filter_prompts(
        self,
        set_name: str,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> list[Prompt]:
        """Test setindeki prompt'ları kategori ve/veya zorluk seviyesine göre filtrele.

        Args:
            set_name: Filtrelenecek test seti adı.
            category: Filtrelenecek kategori değeri (ör. ``"translation"``).
            difficulty: Filtrelenecek zorluk seviyesi (ör. ``"easy"``).

        Returns:
            Filtre kriterlerini karşılayan prompt listesi.
        """
        test_set = self.load(set_name)
        result = test_set.prompts

        if category is not None:
            result = [p for p in result if p.category.value == category]

        if difficulty is not None:
            result = [p for p in result if p.difficulty.value == difficulty]

        return result

    def serialize(self, test_set: TestSet) -> str:
        """Test setini JSON dizesine serileştir.

        Args:
            test_set: Serileştirilecek test seti.

        Returns:
            Pretty-print JSON dizesi.
        """
        return test_set.model_dump_json(indent=2)

    def deserialize(self, json_str: str) -> TestSet:
        """JSON dizesinden test seti ayrıştır.

        Args:
            json_str: Ayrıştırılacak JSON dizesi.

        Returns:
            Ayrıştırılan ``TestSet`` nesnesi.
        """
        return TestSet.model_validate_json(json_str)
