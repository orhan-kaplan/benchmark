# ComfyUI Video Üretimi

## Yapı

Colab'da ComfyUI + video modelleri serve et. Browser'dan veya Colab proxy ile eriş.

## Notebook'lar

### comfy/colab_ltx_i2v.ipynb — LTX 2.3

**Workflow toggle:**
```python
WORKFLOW = 'i2v'  # 'i2v' | 'ic_control' | 'id_lora'
```

| Workflow | Ne yapar |
|----------|----------|
| `i2v` | Image → Video (tek image'dan video üret) |
| `ic_control` | Video → Video (kontrol videosu + referans image → yeni video) |
| `id_lora` | Talking head (yüz + ses → senkronlu video) |

**Modeller (workflow'a göre otomatik yönetim):**

Ortak (silinmez):
- `gemma_3_12B_it_fp4_mixed.safetensors` (text encoder)
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (upscaler)
- `LTX23_video_vae_bf16.safetensors` (VAE)
- `ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors` (distilled LoRA)

Workflow'a özel (diğerleri silinir):
- i2v: `ltx-2.3-22b-dev-fp8.safetensors`
- ic_control: `ltx-2.3-22b-distilled-fp8.safetensors` + `moge_2_vitl_normal_fp16.safetensors` + control LoRA
- id_lora: `ltx-2.3-22b-dev-fp8.safetensors` + `ltx-2.3-id-lora-talkvid-3k.safetensors`

**Custom Node'lar:**
- ComfyUI-LTXVideo (Lightricks resmi)
- ComfyUI-VideoHelperSuite
- ComfyUI-Frame-Interpolation
- ComfyUI-VideoUpscale_WithModel
- ComfyUI-Manager
- ComfyUI-Impact-Pack
- rgthree-comfy

---

### comfy/colab_wan_i2v.ipynb — WAN 2.2

Tek workflow: I2V (Image-to-Video, hassas hareket kontrolü).

**Modeller:**
- `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`
- `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`
- LightX2V 4-step LoRA'lar (hız)
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (text encoder)
- `wan_2.1_vae.safetensors`
- `clip_vision_h.safetensors`

---

### comfy/colab_wan_animate.ipynb — WAN 2.2 Animate Character Swap

Kaynak videodan karakter hareketini al, hedef karaktere uygula. Pose + face detection + SAM2 masking tabanlı.

**Pipeline:**
```
Input Video → Pose/Face Detection → SAM2 Masking → WAN 2.2 Animate 14B → Character Swap Video
```

**Modeller:**
- `Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors` (ana model, Kijai FP8)
- `WanAnimate_relight_lora_fp16.safetensors` (aydınlatma uyumu LoRA)
- `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` (hız LoRA)
- `umt5-xxl-enc-bf16.safetensors` (text encoder, bf16)
- `wan_2.1_vae.safetensors` (VAE)
- `clip_vision_h.safetensors` (CLIP Vision)
- `sam2.1_hiera_base_plus.safetensors` (SAM2 segmentasyon, otomatik indirilir)
- `vitpose-l-wholebody.onnx` (pose detection)
- `yolov10m.onnx` (person detection)

**Custom Node'lar:**
- ComfyUI-WanVideoWrapper (kijai) — model yükleme, sampling, decode
- ComfyUI-KJNodes (kijai) — image/mask utility
- ComfyUI-VideoHelperSuite (Kosinkadink) — video I/O
- ComfyUI-segment-anything-2 (kijai) — SAM2 segmentasyon
- ComfyUI-WanAnimatePreprocess (kijai) — ViTPose + YOLO pose/face detection

**Kullanım:**
1. A-B-C hücrelerini sırayla çalıştır
2. ComfyUI açılınca `Wan 2.2 Animate Character Swap.json` workflow'unu yükle
3. Input video + referans karakter görseli ayarla → Queue Prompt

## Tunnel Seçimi

```python
USE_CLOUDFLARE = False  # False=Colab proxy (hızlı) | True=Cloudflare (sabit URL, yavaş UI)
```

- Colab proxy: Hızlı UI tepkisi, URL her session'da değişir
- Cloudflare: `comfyui.ersamely.com`, yavaş (latency) ama sabit

## Disk Yönetimi

- Workflow değiştirip B çalıştırınca → eski modeller silinir, yenileri indirilir
- Ortak modeller yerinde kalır (tekrar indirilmez)
- Session kapanınca tüm disk silinir (kalıcılık yok — Drive opsiyonel)
