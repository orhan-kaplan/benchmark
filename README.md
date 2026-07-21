# LLM Inference Platform

LLM modellerini sunmak için modüler bir platform.

## Proje Yapısı

```
├── api/                    # FastAPI API Gateway (vLLM proxy, auth, rate limiting)
├── scripts/                # Yardımcı betikler
├── colab_vllm.ipynb        # Colab notebook (vLLM backend)
├── colab_gguf.ipynb        # Colab notebook (llama.cpp GGUF backend)
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

1. `colab_vllm.ipynb` dosyasını Google Colab'a yükle
2. Colab Secrets'e `HF_TOKEN` ve `CF_TUNNEL_TOKEN` ekle
3. Hücreleri sırayla çalıştır
4. `https://api.ersamely.com` üzerinden erişim hazır

### 3. Open WebUI'ı Colab'a bağla

Open WebUI → Settings → Connections → OpenAI API:
- URL: `https://api.ersamely.com/v1`
- API Key: Colab Secrets'teki `VLLM_API_KEY` değeri

## Tunnel (Cloudflare)

Named tunnel ile sabit URL: `https://api.ersamely.com`
- Hesap ve domain Cloudflare'da kayıtlı
- Rate limit yok, session timeout yok
- API key ile korumalı (yetkisiz erişim engellenir)
- Colab Secrets'e `CF_TUNNEL_TOKEN` ve `VLLM_API_KEY` ekle

## API Gateway

FastAPI tabanlı OpenAI-uyumlu proxy katmanı. vLLM backend'lerine istekleri yönlendirir.

### Endpoint'ler

- `POST /v1/chat/completions` — Chat completion (streaming destekli)
- `POST /v1/completions` — Text completion
- `GET /v1/models` — Kayıtlı modelleri listele
- `GET /health` — Sağlık kontrolü

### Özellikler

- Auth (API key doğrulama)
- Rate limiting
- Request logging
- Multi-model routing
- Test modu (mock yanıt)

## Testler

```bash
# API testleri
python -m pytest api/tests/ -v
```

## Gereksinimler

- Python 3.11+
- Docker & Docker Compose
- Google Colab (GPU modelleri için)
