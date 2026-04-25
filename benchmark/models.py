"""Benchmark sistemi Pydantic veri modelleri."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, field_validator


# --- Enum'lar ---


class Backend(str, Enum):
    VLLM = "vllm"
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"


class PromptCategory(str, Enum):
    TRANSLATION = "çeviri"
    CODING = "kodlama"
    REASONING = "muhakeme"
    CREATIVE = "yaratıcı_yazarlık"
    GENERAL = "genel_bilgi"


class Difficulty(str, Enum):
    EASY = "kolay"
    MEDIUM = "orta"
    HARD = "zor"


# --- Model Kataloğu ---


class ModelEntry(BaseModel):
    """Tek bir model kaydı."""

    name: str
    repo: str
    format: str
    quantization: Optional[str] = None
    backend: Backend
    tags: list[str] = []
    api_endpoint: str


# --- Test Seti ---


class Prompt(BaseModel):
    """Tek bir test prompt'u."""

    id: str
    category: PromptCategory
    subcategory: Optional[str] = None
    text: str
    difficulty: Difficulty = Difficulty.MEDIUM
    reference_answer: Optional[str] = None


class TestSet(BaseModel):
    """Bir test seti koleksiyonu."""

    name: str
    description: Optional[str] = None
    prompts: list[Prompt] = []


# --- Üretim Parametreleri ---


class GenerationParams(BaseModel):
    """LLM üretim parametreleri."""

    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    repetition_count: int = 1


# --- Metrikler ---


class ResponseMetrics(BaseModel):
    """Tek bir yanıt için performans metrikleri."""

    ttft_ms: Optional[float] = None
    total_time_ms: float
    tokens_per_second: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class VRAMSnapshot(BaseModel):
    """Tek bir VRAM ölçümü."""

    timestamp: datetime
    used_mb: float
    total_mb: float
    source: str = "nvidia_smi"


# --- Sonuçlar ---


class PromptResult(BaseModel):
    """Tek bir prompt-model çifti sonucu."""

    prompt_id: str
    model_name: str
    response_text: Optional[str] = None
    metrics: Optional[ResponseMetrics] = None
    vram_snapshot: Optional[VRAMSnapshot] = None
    error: Optional[str] = None
    success: bool = True


class BenchmarkRunConfig(BaseModel):
    """Benchmark çalıştırma yapılandırması."""

    test_set_name: str
    model_names: list[str]
    params: GenerationParams = GenerationParams()


class BenchmarkRun(BaseModel):
    """Tam bir benchmark çalıştırması."""

    run_id: str
    timestamp: datetime
    config: BenchmarkRunConfig
    results: list[PromptResult] = []
    completed: bool = False


# --- Puanlama ---


class ScoreEntry(BaseModel):
    """Tek bir puan kaydı."""

    prompt_id: str
    model_name: str
    manual_score: Optional[int] = None
    judge_score: Optional[float] = None
    comment: Optional[str] = None

    @field_validator("manual_score")
    @classmethod
    def validate_manual_score(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1 or v > 10):
            raise ValueError("manual_score must be between 1 and 10")
        return v


class JudgeConfig(BaseModel):
    """Yargıcı model yapılandırması."""

    model_name: str
    api_endpoint: str
    prompt_templates: dict[str, str] = {
        "çeviri": (
            "Aşağıdaki çeviriyi değerlendir. Doğruluk (1-5) ve akıcılık (1-5) olarak puanla.\n"
            "Kaynak dil: Çince, hedef dil: İngilizce.\n"
            "Toplam puan = doğruluk + akıcılık (2-10 arası).\n"
            "Sadece toplam puanı yaz.\n\n"
            "Kaynak metin: {question}\n"
            "Çeviri: {answer}\n"
            "Puan:"
        ),
        "kodlama": (
            "Aşağıdaki kod yanıtını değerlendir. Çalışırlık (1-5) ve kod kalitesi (1-5) olarak puanla.\n"
            "Toplam puan = çalışırlık + kalite (2-10 arası).\n"
            "Sadece toplam puanı yaz.\n\n"
            "Soru: {question}\n"
            "Yanıt: {answer}\n"
            "Puan:"
        ),
        "default": (
            "Aşağıdaki yanıtı 1-10 arası puanla. Doğruluk, tamlık ve netlik kriterlerini kullan.\n"
            "Sadece sayıyı yaz.\n\n"
            "Soru: {question}\n"
            "Yanıt: {answer}\n"
            "Puan:"
        ),
    }


# --- API Yanıt ---


class APIResponse(BaseModel):
    """API Client'ın döndürdüğü ham yanıt."""

    status_code: int
    response_text: Optional[str] = None
    usage: Optional[dict] = None
    error: Optional[str] = None
    elapsed_ms: float
    ttft_ms: Optional[float] = None


# --- Raporlama ---


class MatrixCell(BaseModel):
    """Karşılaştırma matrisinin tek bir hücresi."""

    avg_score: Optional[float] = None
    avg_time_ms: Optional[float] = None
    sample_count: int = 0


class ComparisonMatrix(BaseModel):
    """Model × Kategori karşılaştırma matrisi."""

    models: list[str]
    categories: list[str]
    cells: dict[str, dict[str, MatrixCell]]


class ModelRanking(BaseModel):
    """Tek bir model sıralaması."""

    model_name: str
    rank: int
    avg_score: Optional[float] = None
    avg_time_ms: Optional[float] = None


class VRAMReport(BaseModel):
    """VRAM kullanım raporu."""

    models_within_limit: list[dict]
    models_exceeding_limit: list[dict]
    vram_limit_mb: Optional[int] = None


class RunSummary(BaseModel):
    """Çalıştırma özet raporu."""

    run_id: str
    timestamp: datetime
    total_prompts: int
    total_models: int
    successful_results: int
    failed_results: int
    matrix: ComparisonMatrix
