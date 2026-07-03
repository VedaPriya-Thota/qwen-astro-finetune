# Deploying the Fine-Tuned Vedaz Model on a VPS using vLLM

## 1. Objective

This document describes the process of deploying the LoRA-fine-tuned
Qwen2.5-0.5B-Instruct "Vedaz" model on a VPS, served through vLLM as an
OpenAI-compatible API endpoint.

## 2. VPS Provisioning

**Important note on hardware:** vLLM's core value (PagedAttention, continuous
batching, high-throughput serving) is built around and optimized for NVIDIA
GPUs. vLLM does have a CPU backend, but it is significantly slower and less
mature than the CUDA path, and is not recommended for real production
traffic. For an honest, production-oriented deployment, the recommendation
below is a small GPU VPS. A CPU-only VPS is covered as a fallback for
low-traffic/demo scenarios.

**Recommended (GPU path):**
- 1x NVIDIA GPU with at least 8-16GB VRAM (e.g., an RTX 4000-class or
  A10G-class instance) — comfortably covers a merged 0.5B model plus KV
  cache headroom for concurrent requests
- 4 vCPU, 16GB system RAM
- 40GB+ SSD storage (base model + adapter + OS + vLLM install)
- Ubuntu 22.04 LTS

**Fallback (CPU path, low traffic only):**
- 4-8 vCPU, 16GB RAM minimum
- Same storage/OS requirements
- Expect noticeably higher per-request latency than GPU

## 3. Environment Setup

```bash
# System update and base packages
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git build-essential

# (GPU path only) NVIDIA driver + CUDA toolkit
sudo apt install -y nvidia-driver-535
sudo reboot
nvidia-smi   # verify GPU is detected after reboot

# Create an isolated environment for serving
python3.11 -m venv /opt/vedaz-serve/venv
source /opt/vedaz-serve/venv/bin/activate

# Install vLLM
# GPU build (CUDA 12.1, matches most current-gen GPU VPS images):
pip install vllm

# CPU-only build instead, use:
# pip install vllm --extra-index-url https://download.pytorch.org/whl/cpu
```

## 4. Model Packaging: Merge the LoRA Adapter

vLLM can serve LoRA adapters natively via `--enable-lora`, or you can merge
the adapter into the base model weights ahead of time for simpler serving.
For a single, fixed adapter (our case — one persona, not swapping between
multiple adapters at runtime), **merging is simpler and recommended**.

```python
# merge_adapter.py — run once, locally or on the VPS, before deployment
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "./outputs/adapter/final"
merged_output_path = "./outputs/merged-model"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id)

model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()   # folds LoRA weights into the base weights

model.save_pretrained(merged_output_path)
tokenizer.save_pretrained(merged_output_path)
```

Upload `outputs/merged-model/` to the VPS (e.g., via `scp` or `rsync`):

```bash
rsync -avz ./outputs/merged-model/ user@your-vps-ip:/opt/vedaz-serve/model/
```

**Alternative — serve base + adapter separately** (useful if you expect to
swap/add adapters later without re-merging):

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --enable-lora \
    --lora-modules vedaz=/opt/vedaz-serve/adapter
```

## 5. Launching the vLLM Server

```bash
source /opt/vedaz-serve/venv/bin/activate

vllm serve /opt/vedaz-serve/model \
    --served-model-name vedaz-astrologer \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85
```

Key flags explained:
- `--served-model-name`: the model identifier clients will use in API calls
- `--dtype float16`: halves memory footprint vs float32 on GPU with
  negligible quality loss at this model size
- `--max-model-len`: caps context length; matches training's `max_seq_length`
  headroom, keeps KV-cache memory bounded
- `--gpu-memory-utilization`: fraction of VRAM vLLM is allowed to pre-allocate
  for KV cache — 0.85 leaves headroom for the OS/driver

CPU fallback (drop GPU-specific flags):
```bash
vllm serve /opt/vedaz-serve/model \
    --served-model-name vedaz-astrologer \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float32 \
    --max-model-len 2048
```

## 6. Process Management — systemd Service

Running `vllm serve` directly in a terminal dies when the SSH session ends.
Use a systemd unit so it survives disconnects and reboots:

```ini
# /etc/systemd/system/vedaz-vllm.service
[Unit]
Description=Vedaz vLLM Inference Server
After=network.target

[Service]
User=vedaz
WorkingDirectory=/opt/vedaz-serve
ExecStart=/opt/vedaz-serve/venv/bin/vllm serve /opt/vedaz-serve/model \
    --served-model-name vedaz-astrologer \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 2048
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable vedaz-vllm
sudo systemctl start vedaz-vllm
sudo systemctl status vedaz-vllm     # verify it's running
journalctl -u vedaz-vllm -f          # tail logs
```

## 7. Reverse Proxy (Nginx)

Exposes the API on standard port 443 with TLS, rather than raw port 8000:

```nginx
# /etc/nginx/sites-available/vedaz-api
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;   # generation can take a while
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/vedaz-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# TLS via Let's Encrypt
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com
```

## 8. Testing the Deployment

vLLM exposes an OpenAI-compatible endpoint out of the box:

```bash
curl http://api.yourdomain.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vedaz-astrologer",
    "messages": [
      {"role": "system", "content": "You are Vedaz'\''s AI Vedic astrologer. You never predict death, illness, or guaranteed misfortune."},
      {"role": "user", "content": "Will I become rich this year?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

Expected response shape matches OpenAI's chat completions format
(`choices[0].message.content`), which means any existing OpenAI-client
codebase can point at this endpoint with just a `base_url` change.

## 9. Security & Operational Notes

- **Firewall**: only expose ports 80/443 publicly; keep 8000 bound to
  localhost and reachable only through Nginx (`ufw allow 80,443/tcp`,
  `ufw deny 8000/tcp` from external)
- **API authentication**: vLLM supports an `--api-key` flag to require a
  bearer token on requests — recommended for anything beyond local testing
- **Monitoring**: `journalctl -u vedaz-vllm`, plus `nvidia-smi` (GPU) or
  `htop` (CPU) for resource usage; vLLM also exposes a `/metrics` endpoint
  (Prometheus format) for request latency/throughput
- **Rate limiting / concurrency**: vLLM's continuous batching handles
  multiple concurrent requests efficiently on GPU; on CPU, concurrent
  requests will queue and serialize much more noticeably — plan capacity
  accordingly

## 10. Known Limitations

- CPU inference latency is materially higher than GPU (multi-second per
  response vs. sub-second) — acceptable for a demo/assessment, not for
  production chat traffic
- This deployment serves a single merged model/adapter; multi-tenant
  adapter-swapping (`--enable-lora` with multiple `--lora-modules`) is
  supported by vLLM but adds complexity not needed for this assignment's
  scope
- No autoscaling/load balancing is covered here — single-instance
  deployment is appropriate for the assessment's scope; a real production
  rollout would sit this behind a load balancer with multiple replicas