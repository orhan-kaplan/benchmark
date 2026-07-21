# Japonca Altyazı (Whisper)

## Yapı

Japonca ses/video → SRT altyazı. Tek notebook, config toggle'ları ile farklı kombinasyonlar.

## Notebook: colab_whisper.ipynb

### Config

```python
WHISPER_MODEL       = 'large-v3'     # 'large-v3' | 'RoachLin/kotoba-whisper-v2.2-faster'
USE_VOICE_ISOLATION = True           # Demucs htdemucs_ft ile vocal izolasyon
TIMESTAMP_METHOD    = 'stable-ts'    # 'stable-ts' | 'none'
USE_LLM_REFINEMENT  = False          # Harici LLM ile düzeltme
```

### Pipeline

```
Video → ffmpeg (16kHz mono) → [Demucs vocal] → Whisper/stable-ts → [LLM düzeltme] → SRT
```

### Araştırma Sonuçları

- **large-v3** Japonca'da en iyi sonucu veriyor (anime dahil)
- **kotoba-whisper** (CTranslate2 dönüşümü) kalite kaybı yaşadı — community dönüşüm sorunlu
- **Demucs vocal izolasyon** anime/müzikli içerikte belirgin iyileşme sağlıyor
- **stable-ts** Japonca'da güvenilir (cross-attention tabanlı, dil bağımsız)
- **WhisperX** Blackwell Colab ortamıyla uyumsuz (torchvision/scipy çakışması)
- **ReazonSpeech-NeMo** Colab'da çalışmıyor (NeMo numpy>=2 uyumsuzluğu — GitHub issue #14505)

### LLM Düzeltme (opsiyonel)

Harici API kullanır (vLLM tunnel, OpenAI, vb.):
```python
LLM_API_URL  = 'https://api.ersamely.com/v1'
LLM_API_KEY  = ''  # Colab Secrets'ten VLLM_API_KEY
LLM_MODEL    = 'Qwen/Qwen3.6-35B-A3B-FP8'
```

### Compare Notebook: colab_whisper_compare.ipynb

Tek çalıştırmada 4 SRT üretir (vocal_iso × timestamp_method matrix). large-v3 ile.
