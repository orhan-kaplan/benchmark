# LoRA Eğitimi

## Notebook'lar (oluşturulacak)

- `comfy/colab_lora_flux.ipynb` — FLUX.2 LoRA (Ostris AI-Toolkit)
- `comfy/colab_lora_sdxl.ipynb` — SDXL LoRA (Kohya SS)

## FLUX.2 LoRA Eğitimi

### Araç: Ostris AI-Toolkit
- GitHub: https://github.com/ostris/ai-toolkit
- FLUX.2-dev ve FLUX.2-klein native desteği
- Tek YAML config dosyası
- Colab uyumlu

### Dataset Gereksinimleri

| Tip | Görsel sayısı | Açıklama |
|-----|--------------|----------|
| Karakter | 15-30 | Farklı açı, ifade, ışık, kıyafet |
| Stil | 20-50 | Aynı stilde farklı konular |
| Gerçek kişi | 15-20 | Tanınabilir, net, yüksek kalite |

### Dataset Kalite Kuralları
1. Eğitmek istediğin şeyde tutarlı, geri kalanda çeşitli
2. Bulanık/watermark'lı görsel ekleme — tek kötü görsel sonucu bozar
3. Her görsele detaylı caption + trigger word (ör: `ohk_person`)
4. Çözünürlük: 1024x1024 minimum, 1024x1536 dikey karakter için ideal
5. Fotoğraftaki kişiyi tanıyamıyorsan dataset'ten çıkar
6. 50'den fazla görsel genelde gereksiz (overfitting riski)

### Eğitim Parametreleri (96GB VRAM)

| Parametre | Değer |
|-----------|-------|
| Model | FLUX.2-dev (bf16, quantize gerekmez) |
| Steps | 3000-5000 |
| Learning rate | 1e-4 |
| LoRA rank | 16-32 |
| Epochs | 10-15 |
| Batch size | 4-8 (96GB VRAM avantajı) |
| Precision | bf16 (full, quantize yok) |
| Süre | ~1-2 saat (30 görsel) |

### Captioning

Otomatik: Florence-2 veya BLIP-2
Manuel: Daha kaliteli sonuç, önerilir

### Çıktı

`.safetensors` dosyası → `ComfyUI/models/loras/` klasörüne koy → workflow'da kullan

---

## SDXL LoRA Eğitimi

### Araç: Kohya SS (sd-scripts)
- GitHub: https://github.com/kohya-ss/sd-scripts
- SDXL için en olgun araç
- Community preset'leri çok

### Detaylar

- Network Rank: 32-64
- Network Alpha: rank/2
- Learning Rate UNet: 1e-4
- Learning Rate Text Encoder: 5e-5
- Optimizer: Prodigy (otomatik LR) veya AdamW8bit
- Scheduler: cosine_with_restarts
- Batch size: 4-8 (96GB VRAM)
- Epochs: 10-20
- Precision: bf16
- Süre: ~20-40 dakika (30 görsel, 96GB VRAM)

### SDXL Özel Notlar
- `enable_bucket` — farklı aspect ratio desteği
- `cache_latents` + `cache_latents_to_disk` — hız artırır
- Prodigy optimizer LR'yi otomatik bulur
- Caption dropout %5 — genelleştirme için
- Text Encoder eğitimi önerilir (FLUX'tan farklı)

---

## Notlar

- 96GB VRAM ile bf16 full precision eğitim → quantize gerekmez, kalite maksimum
- FLUX.2-dev 32B parametre — 24GB kartlarda FP8 gerekir, senin 96GB'de gerek yok
- Dataset hazırlık eğitimden daha önemli — iyi dataset + basit config > kötü dataset + mükemmel config
