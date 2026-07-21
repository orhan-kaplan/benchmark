# Proje Mimarisi

## Genel Bakış

Bu proje, Google Colab GPU'larını kullanarak AI modellerini serve eden ve Cloudflare Tunnel ile dışarıya açan bir platform. Üç ana alan: LLM inference, ses/altyazı işleme, görsel/video üretimi.

## Altyapı

```
[Google Colab GPU] → [Cloudflare Tunnel] → [api.ersamely.com / comfyui.ersamely.com]
                                          ↓
                                   [Lokal browser / Open WebUI]
```

- **Domain:** ersamely.com (Cloudflare DNS, Turkticaret registrar)
- **Tunnel:** Named tunnel "ersamely" — tek token ile birden fazla subdomain
- **GPU:** NVIDIA RTX PRO 6000 Blackwell (96GB VRAM) veya A100
- **Güvenlik:** vLLM → API key, ComfyUI → Cloudflare Access (email OTP)

## Dizin Yapısı

```
llm-ui/
├── api/                          # FastAPI API Gateway (vLLM proxy)
│   ├── app/
│   │   ├── dependencies.py       # DI providers (get_config, get_registry)
│   │   ├── main.py               # FastAPI app, middleware registration
│   │   ├── middleware/           # auth, rate_limiter, logging
│   │   ├── models/              # Pydantic request/response/config
│   │   ├── routers/             # chat, completions, models, health
│   │   └── services/            # model_registry, transformer, health
│   ├── tests/                   # 127 test (pytest)
│   ├── Dockerfile               # Non-root user, healthcheck
│   ├── .dockerignore
│   ├── requirements.txt         # Pinned versions
│   └── requirements-dev.txt     # Test deps
├── comfy/                        # ComfyUI Colab notebook'ları
│   ├── colab_ltx_i2v.ipynb      # LTX 2.3 video (I2V, IC-Control, ID-LoRA)
│   ├── colab_wan_i2v.ipynb      # WAN 2.2 video (I2V)
│   └── colab_image_edit.ipynb   # Görsel düzenleme (Qwen + FLUX.2)
├── colab_vllm.ipynb             # vLLM model serve + tunnel
├── colab_gguf.ipynb             # llama.cpp GGUF serve + tunnel
├── colab_whisper.ipynb          # Japonca altyazı (Whisper + Demucs)
├── colab_whisper_compare.ipynb  # Whisper A/B test (large-v3)
├── colab_reazonspeech.ipynb     # ReazonSpeech (deneysel, NeMo uyumsuz)
├── docker-compose.yml           # Open WebUI + API Gateway + vLLM
├── scripts/                     # start.sh, health_check.sh
├── .env                         # Ortam değişkenleri (git'te yok)
├── .gitignore
├── pyproject.toml
└── README.md
```

## Cloudflare Tunnel Yapısı

Tek tunnel (`ersamely`), birden fazla hostname:

| Subdomain | Port | Kullanım |
|-----------|------|----------|
| api.ersamely.com | 8090 | vLLM / llama.cpp inference |
| comfyui.ersamely.com | 8188 | ComfyUI web arayüzü |

## Colab Secrets (tüm notebook'lar için)

| Secret | Kullanım |
|--------|----------|
| `HF_TOKEN` | HuggingFace model indirme |
| `CF_TUNNEL_TOKEN` | Cloudflare named tunnel |
| `VLLM_API_KEY` | vLLM API erişim koruması |
