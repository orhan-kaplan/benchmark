"""Benchmark CLI — argparse tanımları ve komut yönlendirme.

Kullanım:
    python -m benchmark run --test-set <name> --models <m1,m2>
    python -m benchmark resume --run-id <id>
    python -m benchmark models list|add|remove
    python -m benchmark test-sets list|create
    python -m benchmark score --run-id <id>
    python -m benchmark report --run-id <id>
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from benchmark.api_client import APIClient
from benchmark.catalog import ModelCatalog
from benchmark.metrics import MetricsCollector
from benchmark.models import (
    Backend,
    BenchmarkRunConfig,
    GenerationParams,
    JudgeConfig,
    ModelEntry,
    TestSet,
)
from benchmark.reporter import ReportGenerator
from benchmark.runner import BenchmarkRunner
from benchmark.scoring import ScoringEngine
from benchmark.storage import StorageManager
from benchmark.test_sets import TestSetManager

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("benchmark_data")
DEFAULT_CONFIG_PATH = DEFAULT_DATA_PATH / "config.json"


def _load_default_config() -> dict:
    """Load default config from benchmark_data/config.json if it exists."""
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _build_generation_params(args: argparse.Namespace) -> GenerationParams:
    """Build GenerationParams from default config + CLI overrides."""
    config = _load_default_config()
    params_cfg = config.get("params", {})

    temperature = args.temperature if args.temperature is not None else params_cfg.get("temperature", 0.7)
    max_tokens = args.max_tokens if args.max_tokens is not None else params_cfg.get("max_tokens", 1024)
    top_p = args.top_p if args.top_p is not None else params_cfg.get("top_p", 1.0)
    repeat = args.repeat if args.repeat is not None else params_cfg.get("repetition_count", 1)

    return GenerationParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        repetition_count=repeat,
    )


# ---------------------------------------------------------------------------
# Handler'lar — bileşenlere bağlı gerçek uygulamalar
# ---------------------------------------------------------------------------


def handle_run(args: argparse.Namespace) -> None:
    """Benchmark çalıştırma komutu."""
    storage = StorageManager(DEFAULT_DATA_PATH)
    catalog = ModelCatalog(DEFAULT_DATA_PATH)
    test_set_manager = TestSetManager(DEFAULT_DATA_PATH)
    api_client = APIClient()
    metrics_collector = MetricsCollector()

    runner = BenchmarkRunner(
        catalog=catalog,
        test_set_manager=test_set_manager,
        api_client=api_client,
        metrics_collector=metrics_collector,
        storage=storage,
    )

    model_names = [m.strip() for m in args.models.split(",")]
    params = _build_generation_params(args)

    config = BenchmarkRunConfig(
        test_set_name=args.test_set,
        model_names=model_names,
        params=params,
    )

    print(f"Benchmark başlatılıyor: test_set={args.test_set}, models={model_names}")
    result = asyncio.run(runner.run(config))
    print(f"Benchmark tamamlandı: run_id={result.run_id}, "
          f"sonuç sayısı={len(result.results)}")

    asyncio.run(api_client.close())


def handle_resume(args: argparse.Namespace) -> None:
    """Yarıda kalan benchmark çalıştırmasını devam ettir."""
    storage = StorageManager(DEFAULT_DATA_PATH)
    catalog = ModelCatalog(DEFAULT_DATA_PATH)
    test_set_manager = TestSetManager(DEFAULT_DATA_PATH)
    api_client = APIClient()
    metrics_collector = MetricsCollector()

    runner = BenchmarkRunner(
        catalog=catalog,
        test_set_manager=test_set_manager,
        api_client=api_client,
        metrics_collector=metrics_collector,
        storage=storage,
    )

    print(f"Çalıştırma devam ettiriliyor: run_id={args.run_id}")
    result = asyncio.run(runner.resume(args.run_id))
    print(f"Çalıştırma tamamlandı: run_id={result.run_id}, "
          f"sonuç sayısı={len(result.results)}")

    asyncio.run(api_client.close())


def handle_models_list(args: argparse.Namespace) -> None:
    """Kayıtlı modelleri listele."""
    catalog = ModelCatalog(DEFAULT_DATA_PATH)
    models = catalog.list_all()

    if not models:
        print("Kayıtlı model bulunamadı.")
        return

    for model in models:
        tags_str = ", ".join(model.tags) if model.tags else "-"
        print(f"  {model.name}  backend={model.backend.value}  "
              f"format={model.format}  endpoint={model.api_endpoint}  "
              f"tags=[{tags_str}]")


def handle_models_add(args: argparse.Namespace) -> None:
    """Yeni model kaydı ekle."""
    catalog = ModelCatalog(DEFAULT_DATA_PATH)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    model = ModelEntry(
        name=args.name,
        repo=args.repo,
        format=args.model_format,
        quantization=args.quantization,
        backend=Backend(args.backend),
        tags=tags,
        api_endpoint=args.endpoint,
    )

    try:
        catalog.register(model)
        print(f"Model kaydedildi: {args.name}")
    except ValueError as e:
        print(f"Hata: {e}", file=sys.stderr)
        sys.exit(1)


def handle_models_remove(args: argparse.Namespace) -> None:
    """Model kaydını sil."""
    catalog = ModelCatalog(DEFAULT_DATA_PATH)

    try:
        catalog.remove(args.name)
        print(f"Model silindi: {args.name}")
    except KeyError as e:
        print(f"Hata: {e}", file=sys.stderr)
        sys.exit(1)


def handle_test_sets_list(args: argparse.Namespace) -> None:
    """Mevcut test setlerini listele."""
    manager = TestSetManager(DEFAULT_DATA_PATH)
    sets = manager.list_sets()

    if not sets:
        print("Mevcut test seti bulunamadı.")
        return

    for name in sorted(sets):
        print(f"  {name}")


def handle_test_sets_create(args: argparse.Namespace) -> None:
    """Yeni test seti oluştur."""
    manager = TestSetManager(DEFAULT_DATA_PATH)

    test_set = TestSet(
        name=args.name,
        description=getattr(args, "description", None),
    )

    manager.create(test_set)
    print(f"Test seti oluşturuldu: {args.name}")


def handle_test_sets_add_prompt(args: argparse.Namespace) -> None:
    """Mevcut bir test setine prompt ekle."""
    from benchmark.models import Difficulty, PromptCategory

    manager = TestSetManager(DEFAULT_DATA_PATH)

    prompt = Prompt(
        id=args.prompt_id,
        category=PromptCategory(args.category),
        subcategory=args.subcategory,
        text=args.text,
        difficulty=Difficulty(args.difficulty),
        reference_answer=args.reference_answer,
    )

    try:
        manager.add_prompt(args.set_name, prompt)
        print(f"Prompt eklendi: {args.prompt_id} → {args.set_name}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Hata: {e}", file=sys.stderr)
        sys.exit(1)


def handle_score(args: argparse.Namespace) -> None:
    """Puanlama komutu."""
    storage = StorageManager(DEFAULT_DATA_PATH)

    if args.auto:
        api_client = APIClient()
        scoring = ScoringEngine(storage=storage, api_client=api_client)

        judge_model_name = args.judge_model
        if not judge_model_name:
            print("Hata: --auto kullanıldığında --judge-model belirtilmelidir.",
                  file=sys.stderr)
            sys.exit(1)

        # Look up judge model endpoint from catalog
        catalog = ModelCatalog(DEFAULT_DATA_PATH)
        judge_entry = catalog.get(judge_model_name)
        if judge_entry is None:
            print(f"Hata: '{judge_model_name}' adında bir model bulunamadı.",
                  file=sys.stderr)
            sys.exit(1)

        judge_config = JudgeConfig(
            model_name=judge_model_name,
            api_endpoint=judge_entry.api_endpoint,
        )

        print(f"Otomatik puanlama başlatılıyor: run_id={args.run_id}, "
              f"judge={judge_model_name}")
        asyncio.run(scoring.auto_score(args.run_id, judge_config))
        print("Otomatik puanlama tamamlandı.")

        asyncio.run(api_client.close())
    else:
        scoring = ScoringEngine(storage=storage)
        unscored = scoring.get_unscored(args.run_id)
        print(f"Puanlanmamış öğe sayısı: {len(unscored)}")
        if unscored:
            print("Puanlanmamış öğeler:")
            for item in unscored:
                print(f"  prompt_id={item.get('prompt_id')}, "
                      f"model={item.get('model_name')}")


def handle_report(args: argparse.Namespace) -> None:
    """Rapor üretme komutu."""
    storage = StorageManager(DEFAULT_DATA_PATH)
    reporter = ReportGenerator(storage=storage)

    run_id = args.run_id
    fmt = args.format

    # Generate summary
    summary = reporter.generate_summary(run_id)
    print(f"Çalıştırma: {summary.run_id}")
    print(f"  Toplam prompt: {summary.total_prompts}, "
          f"Toplam model: {summary.total_models}")
    print(f"  Başarılı: {summary.successful_results}, "
          f"Başarısız: {summary.failed_results}")

    # Export
    output_dir = storage.runs_dir / run_id
    if fmt == "json":
        output_path = output_dir / "report.json"
        reporter.export_json(run_id, output_path)
        print(f"JSON rapor: {output_path}")
    else:
        output_path = output_dir / "report.csv"
        reporter.export_csv(run_id, output_path)
        print(f"CSV rapor: {output_path}")

    # VRAM report if limit specified
    if args.vram_limit is not None:
        vram_report = reporter.generate_vram_report(run_id, args.vram_limit)
        print(f"\nVRAM Analizi (limit: {args.vram_limit} MB):")
        if vram_report.models_within_limit:
            print("  Sığan modeller:")
            for m in vram_report.models_within_limit:
                print(f"    {m['name']}: peak={m['peak_mb']:.0f} MB")
        if vram_report.models_exceeding_limit:
            print("  Sığmayan modeller:")
            for m in vram_report.models_exceeding_limit:
                print(f"    {m['name']}: peak={m['peak_mb']:.0f} MB")


# ---------------------------------------------------------------------------
# argparse yapılandırması
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Ana argparse parser'ını ve tüm alt komutları oluşturur."""

    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="LLM Benchmark Sistemi — modelleri test et, karşılaştır ve raporla.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Alt komutlar")

    # --- run ---
    run_parser = subparsers.add_parser(
        "run",
        help="Benchmark çalıştırması başlat",
        description="Belirtilen test setini seçilen modellere gönderir ve sonuçları toplar.",
    )
    run_parser.add_argument(
        "--test-set", required=True, help="Kullanılacak test seti adı"
    )
    run_parser.add_argument(
        "--models",
        required=True,
        help="Virgülle ayrılmış model adları (örn: qwen,gemma)",
    )
    run_parser.add_argument(
        "--temperature", type=float, default=None, help="Üretim sıcaklığı (varsayılan: config)"
    )
    run_parser.add_argument(
        "--max-tokens", type=int, default=None, help="Maksimum üretim token sayısı (varsayılan: config)"
    )
    run_parser.add_argument(
        "--top-p", type=float, default=None, help="Top-p (nucleus) örnekleme değeri (varsayılan: config)"
    )
    run_parser.add_argument(
        "--repeat", type=int, default=None, help="Her prompt'un tekrar sayısı (varsayılan: config)"
    )
    run_parser.set_defaults(func=handle_run)

    # --- resume ---
    resume_parser = subparsers.add_parser(
        "resume",
        help="Yarıda kalan çalıştırmayı devam ettir",
        description="Belirtilen çalıştırma ID'si ile yarıda kalan benchmark'ı kaldığı yerden sürdürür.",
    )
    resume_parser.add_argument(
        "--run-id", required=True, help="Devam ettirilecek çalıştırma ID'si"
    )
    resume_parser.set_defaults(func=handle_resume)

    # --- models (alt komutlu) ---
    models_parser = subparsers.add_parser(
        "models",
        help="Model kataloğu yönetimi",
        description="Kayıtlı modelleri listele, ekle veya sil.",
    )
    models_sub = models_parser.add_subparsers(dest="models_command", help="Model alt komutları")

    # models list
    models_list_parser = models_sub.add_parser("list", help="Kayıtlı modelleri listele")
    models_list_parser.set_defaults(func=handle_models_list)

    # models add
    models_add_parser = models_sub.add_parser("add", help="Yeni model kaydı ekle")
    models_add_parser.add_argument("--name", required=True, help="Model adı")
    models_add_parser.add_argument("--repo", required=True, help="HuggingFace repo yolu")
    models_add_parser.add_argument(
        "--format", dest="model_format", default="FP16", help="Model formatı (varsayılan: FP16)"
    )
    models_add_parser.add_argument("--quantization", default=None, help="Quantization türü")
    models_add_parser.add_argument(
        "--backend", required=True, choices=["vllm", "ollama", "llama_cpp"], help="Backend türü"
    )
    models_add_parser.add_argument("--tags", default="", help="Virgülle ayrılmış etiketler")
    models_add_parser.add_argument("--endpoint", required=True, help="API endpoint URL'si")
    models_add_parser.set_defaults(func=handle_models_add)

    # models remove
    models_remove_parser = models_sub.add_parser("remove", help="Model kaydını sil")
    models_remove_parser.add_argument("--name", required=True, help="Silinecek model adı")
    models_remove_parser.set_defaults(func=handle_models_remove)

    # --- test-sets (alt komutlu) ---
    ts_parser = subparsers.add_parser(
        "test-sets",
        help="Test seti yönetimi",
        description="Test setlerini listele veya yeni test seti oluştur.",
    )
    ts_sub = ts_parser.add_subparsers(dest="test_sets_command", help="Test seti alt komutları")

    # test-sets list
    ts_list_parser = ts_sub.add_parser("list", help="Mevcut test setlerini listele")
    ts_list_parser.set_defaults(func=handle_test_sets_list)

    # test-sets create
    ts_create_parser = ts_sub.add_parser("create", help="Yeni test seti oluştur")
    ts_create_parser.add_argument("--name", required=True, help="Test seti adı")
    ts_create_parser.add_argument("--description", default=None, help="Test seti açıklaması")
    ts_create_parser.set_defaults(func=handle_test_sets_create)

    # test-sets add-prompt
    ts_add_parser = ts_sub.add_parser("add-prompt", help="Mevcut test setine prompt ekle")
    ts_add_parser.add_argument("--set-name", required=True, help="Hedef test seti adı")
    ts_add_parser.add_argument("--prompt-id", required=True, help="Prompt benzersiz kimliği")
    ts_add_parser.add_argument(
        "--category", required=True,
        choices=["translation", "coding", "reasoning", "creative", "general"],
        help="Prompt kategorisi",
    )
    ts_add_parser.add_argument("--subcategory", default=None, help="Alt kategori")
    ts_add_parser.add_argument("--text", required=True, help="Prompt metni")
    ts_add_parser.add_argument(
        "--difficulty", default="medium", choices=["easy", "medium", "hard"],
        help="Zorluk seviyesi (varsayılan: orta)",
    )
    ts_add_parser.add_argument("--reference-answer", default=None, help="Referans yanıt")
    ts_add_parser.set_defaults(func=handle_test_sets_add_prompt)

    # --- score ---
    score_parser = subparsers.add_parser(
        "score",
        help="Çalıştırma sonuçlarını puanla",
        description="Belirtilen çalıştırmanın sonuçlarını manuel veya otomatik olarak puanlar.",
    )
    score_parser.add_argument("--run-id", required=True, help="Puanlanacak çalıştırma ID'si")
    score_parser.add_argument(
        "--auto", action="store_true", default=False, help="Otomatik yargıcı model ile puanla"
    )
    score_parser.add_argument(
        "--judge-model", default=None, help="Yargıcı model adı (--auto ile birlikte kullanılır)"
    )
    score_parser.set_defaults(func=handle_score)

    # --- report ---
    report_parser = subparsers.add_parser(
        "report",
        help="Benchmark raporu üret",
        description="Belirtilen çalıştırma için karşılaştırma raporu üretir ve dışa aktarır.",
    )
    report_parser.add_argument("--run-id", required=True, help="Raporlanacak çalıştırma ID'si")
    report_parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Dışa aktarma formatı (varsayılan: json)",
    )
    report_parser.add_argument(
        "--vram-limit", type=int, default=None, help="VRAM limiti (MB) — donanım uyumluluk analizi"
    )
    report_parser.set_defaults(func=handle_report)

    return parser


# ---------------------------------------------------------------------------
# Ana giriş noktası
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI giriş noktası. ``benchmark/__main__.py`` tarafından çağrılır."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # models / test-sets alt komut kontrolü
    if args.command == "models" and getattr(args, "models_command", None) is None:
        parser.parse_args(["models", "--help"])
    if args.command == "test-sets" and getattr(args, "test_sets_command", None) is None:
        parser.parse_args(["test-sets", "--help"])

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)
