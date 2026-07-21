# Cloudflare Tunnel Kurulumu

## Domain

- Domain: `ersamely.com`
- Registrar: Turkticaret
- DNS: Cloudflare (NS: cullen.ns.cloudflare.com, jewel.ns.cloudflare.com)

## Tunnel

- Tunnel adı: `ersamely`
- Tunnel ID: `5cc372cf-244c-4511-a8a9-a23a06e627f8`
- Token: Colab Secrets'te `CF_TUNNEL_TOKEN` olarak saklanır

## Hostname Route'ları

Zero Trust → Networks → Tunnels → ersamely → Public Hostname:

| Hostname | Service | Kullanım |
|----------|---------|----------|
| api.ersamely.com | HTTP://localhost:8090 | vLLM API |
| comfyui.ersamely.com | HTTP://localhost:8188 | ComfyUI |

## Cloudflare Access

Zero Trust → Access → Applications:
- `comfyui.ersamely.com` — Email OTP doğrulaması
- `api.ersamely.com` — API key ile korumalı (Access gerekmez)

## Colab'da Kullanım

```python
token = userdata.get('CF_TUNNEL_TOKEN')
subprocess.Popen(['cloudflared', 'tunnel', '--no-autoupdate', 'run', '--token', token])
```

## Notlar

- Tunnel aynı anda birden fazla hostname serve edebilir
- Session kapanınca tunnel da kapanır (dashboard'da "Down" görünür)
- Route ekleme/silme Colab restart gerektirmez — cloudflared otomatik config güncellemesi alır
- `--no-autoupdate` flag'i önemli — Colab'da auto-update izin sorunu çıkarır
