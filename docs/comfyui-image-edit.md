# ComfyUI Görsel Düzenleme

## Notebook: comfy/colab_image_edit.ipynb

**Workflow toggle:**
```python
WORKFLOW = 'qwen_edit_2511'  # 'qwen_edit_2511' | 'qwen_controlnet' | 'qwen_inpainting' | 'flux2_dev' | 'flux2_klein'
```

## Workflow'lar

| Workflow | Model | Ne yapar |
|----------|-------|----------|
| `qwen_edit_2511` | Qwen Image Edit 2511 | Prompt ile görsel düzenleme |
| `qwen_controlnet` | Qwen 2512 ControlNet Union | Yapı koruyarak yeni görsel (canny, depth, pose, vb.) |
| `qwen_inpainting` | Qwen InstantX Inpainting | Mask ile kısmi düzenleme (obje/arka plan değiştir) |
| `flux2_dev` | FLUX.2-dev 32B | Text-to-image (en kaliteli) + Turbo LoRA |
| `flux2_klein` | FLUX.2-klein 9B | Text-to-image (hızlı) + image edit (base model) |

## Disk Yönetimi

Workflow değiştirip B çalıştırınca:
- Eski workflow'un özel modelleri otomatik silinir
- Ortak modeller (qwen VAE, text encoder) tekrar indirilmez
- Yeni workflow modelleri indirilir

## Model Kaynakları

- Qwen modelleri: `Comfy-Org/Qwen-Image-Edit_ComfyUI`, `Comfy-Org/Qwen-Image_ComfyUI`
- FLUX modelleri: `Comfy-Org/flux2-dev`, `black-forest-labs/FLUX.2-klein-*`
- ControlNet: `alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union`
- Inpainting: `Comfy-Org/Qwen-Image-InstantX-ControlNets`
- Lightning (hız): `lightx2v/Qwen-Image-Edit-2511-Lightning`, `lightx2v/Qwen-Image-Lightning`

## Önemli Notlar

- FLUX.2-dev non-commercial lisanslı. Ticari kullanım için FLUX.2-klein (Apache 2.0) kullan.
- Workflow JSON'ları ComfyUI'ın "Templates" bölümünden veya comfy.org'dan yüklenebilir.
- `checkpoints/` vs `diffusion_models/` — workflow'a göre değişir, notebook doğru klasöre indirir.
