# LLM Inference & Benchmark Platform

LLM modellerini test etmek, karşılaştırmak ve sunmak için modüler bir platform.

## Proje Yapısı

```
├── api/                    # FastAPI API Gateway (vLLM proxy, auth, rate limiting)
├── benchmark/              # Benchmark sistemi (model test & karşılaştırma)
├── benchmark_data/         # Test setleri, model kataloğu, çalıştırma sonuçları
├── tests/                  # Benchmark testleri (190 test)
├── scripts/                # Yardımcı betikler
├── temp.ipynb              # Colab notebook (vLLM + ngrok)
├── docker-compose.yml      # Open WebUI + API Gateway
└── .env                    # Ortam değişkenleri
```

## Hızlı Başlangıç

### 1. Open WebUI'ı başlat (lokal)

```bash
docker compose up -d open-webui
```

Tarayıcıdan: http://localhost:3000

### 2. Colab'da model çalıştır

1. `temp.ipynb` dosyasını Google Colab'a yükle
2. `NGROK_AUTHTOKEN` değerini güncelle
3. Hücreleri sırayla çalıştır
4. ngrok URL'ini al

### 3. Open WebUI'ı Colab'a bağla

Open WebUI → Settings → Connections → OpenAI API:
- URL: `{ngrok_url}/v1`
- API Key: `not-needed`

## Benchmark Sistemi

### Model kataloğu

```bash
# Modelleri listele
python -m benchmark models list

# Model ekle
python -m benchmark models add \
  --name qwen3.6-35b \
  --repo Qwen/Qwen3.6-35B-A3B \
  --backend vllm \
  --endpoint http://localhost:8090

# Model sil
python -m benchmark models remove --name qwen3.6-35b
```

### Test setleri

```bash
# Test setlerini listele
python -m benchmark test-sets list

# Mevcut setler: translation, coding, reasoning, creative, general
```

### Benchmark çalıştır

```bash
# Tek model test
python -m benchmark run \
  --test-set coding \
  --models qwen3.6-35b

# Birden fazla model
python -m benchmark run \
  --test-set translation \
  --models qwen3.6-35b,qwen3.6-27b-fp8

# Özel parametrelerle
python -m benchmark run \
  --test-set reasoning \
  --models qwen3.6-35b \
  --temperature 0.3 \
  --max-tokens 2048

# Yarıda kalan çalıştırmayı devam ettir
python -m benchmark resume --run-id 20260425_143022_a1b2c3
```

### Puanlama

```bash
# Puanlanmamış öğeleri gör
python -m benchmark score --run-id 20260425_143022_a1b2c3

# Otomatik puanlama (judge model ile)
python -m benchmark score \
  --run-id 20260425_143022_a1b2c3 \
  --auto \
  --judge-model qwen3.6-35b
```

### Raporlama

```bash
# JSON rapor
python -m benchmark report --run-id 20260425_143022_a1b2c3

# CSV rapor
python -m benchmark report --run-id 20260425_143022_a1b2c3 --format csv

# VRAM analizi (16GB limit)
python -m benchmark report --run-id 20260425_143022_a1b2c3 --vram-limit 16384
```

## Colab'da Benchmark Çalıştırma

```python
# Colab notebook'unda, vLLM başladıktan sonra:
from benchmark.runner import BenchmarkRunner
from benchmark.catalog import ModelCatalog
from benchmark.test_sets import TestSetManager
from benchmark.api_client import APIClient
from benchmark.metrics import MetricsCollector
from benchmark.storage import StorageManager
from benchmark.models import BenchmarkRunConfig, GenerationParams

storage = StorageManager(Path("benchmark_data"))
catalog = ModelCatalog(Path("benchmark_data"))
test_sets = TestSetManager(Path("benchmark_data"))
api_client = APIClient()
metrics = MetricsCollector()

runner = BenchmarkRunner(catalog, test_sets, api_client, metrics, storage)

config = BenchmarkRunConfig(
    test_set_name="translation",
    model_names=["qwen3.6-35b"],
    params=GenerationParams(temperature=0.7, max_tokens=1024),
)

result = await runner.run(config)
```

## Test Setleri

| Set | Prompt Sayısı | Kategoriler |
|-----|--------------|-------------|
| translation | 10 | Roman, web novel, teknik, günlük (Çince → İngilizce) |
| coding | 10 | Fonksiyon, algoritma, debugging, kod inceleme, JS |
| reasoning | 10 | Matematik, mantık, bilimsel, günlük problem |
| creative | 5 | Hikaye, şiir, senaryo, metin devam ettirme |
| general | 5 | Bilim, teknoloji, tarih, felsefe, pratik |

## Kayıtlı Modeller (Colab)

| Model | Format | Backend |
|-------|--------|---------|
| Qwen/Qwen3.6-35B-A3B | HF | vLLM |
| Qwen/Qwen3.6-35B-A3B-FP8 | FP8 | vLLM |
| Qwen/Qwen3.5-35B-A3B-FP8 | FP8 | vLLM |
| Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 | GPTQ | vLLM |
| Qwen/Qwen3.6-27B | HF | vLLM |
| Qwen/Qwen3.6-27B-FP8 | FP8 | vLLM |
| HauhauCS/Qwen3.5-35B-A3B-Uncensored | HF | vLLM |
| HauhauCS/Qwen3.6-27B-Uncensored | HF | vLLM |
| hesamation/Qwen3.6-35B-A3B-Claude-Distilled | GGUF | vLLM |

## Testler

```bash
# Tüm testler (190 test)
python -m pytest tests/ -v

# Sadece property testleri
python -m pytest tests/test_property_*.py -v

# Sadece birim testleri
python -m pytest tests/ -v --ignore=tests/test_property_*.py
```

## Gereksinimler

- Python 3.11+
- Docker & Docker Compose
- Google Colab (GPU modelleri için)
- ngrok hesabı (Colab tunnel için)
