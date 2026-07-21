# LLM Inference

## Yapı

Colab'da vLLM veya llama.cpp ile model serve et, Cloudflare Tunnel ile `api.ersamely.com` üzerinden eriş.

## Notebook'lar

### colab_vllm.ipynb
- Multi-model switch (MODELS dict'inden seç)
- `--api-key` koruması
- Blackwell/A100 uyumlu
- `VLLM_USE_FLASHINFER=0` (Blackwell flashinfer uyumsuzluğu)

### colab_gguf.ipynb
- llama.cpp ile GGUF model serve
- `--api_key` koruması
- Split GGUF desteği

## Open WebUI Bağlantısı

```bash
docker compose up -d open-webui
```

`.env`'de:
```
VLLM_API_KEY=senin-keyin
```

Open WebUI `https://api.ersamely.com/v1`'e bağlanır. Browser'dan `localhost:3000`.

## Model Kataloğu (colab_vllm.ipynb)

```python
MODELS = {
    'qwen3.6-35b': {'repo': 'Qwen/Qwen3.6-35B-A3B', ...},
    'qwen3.6-35b-fp8': {'repo': 'Qwen/Qwen3.6-35B-A3B-FP8', ...},
    'qwen3-32b': {'repo': 'Qwen/Qwen3-32B', ...},
    ...
}
ACTIVE_MODEL = 'qwen3.6-35b'
```

Model değiştirmek: `ACTIVE_MODEL` değiştir, B bölümünü tekrar çalıştır. Tunnel ayakta kalır.
