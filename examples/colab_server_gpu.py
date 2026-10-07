"""
================================================================================
IMAGEJEV: COLAB NOTEBOOK #1 - GPU REST SERVER (TESLA T4)
================================================================================
Run this in Google Colab (with Runtime -> T4 GPU enabled).

Features:
- Loads OpenCLIP ViT-B-32 into GPU VRAM
- Starts FastAPI + Uvicorn server on port 8000
- Launches Cloudflare Tunnel (cloudflared) for a 100% free, public HTTPS URL
  (No sign-up or auth tokens required!)
================================================================================
"""

# [Cell 1]: Install dependencies & Cloudflare Tunnel
# !pip install -q fastapi uvicorn open-clip-torch torchvision pillow pydantic python-multipart nest-asyncio

import io
import os
import sys
import time
import json
import base64
import threading
import subprocess
import numpy as np
from PIL import Image
import torch
import open_clip

from fastapi import FastAPI, File, Form, UploadFile, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional

print("=" * 70)
print("1. STARTING IMAGEJEV GPU SERVER...")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Device: {gpu_name}")

# Initialize Model & Transforms
model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
model.eval()
tok = open_clip.get_tokenizer("ViT-B-32")
logit_scale = float(model.logit_scale.exp().detach().cpu().item())
text_cache = {}

def get_text_vecs(texts):
    uncached = [t for t in texts if t not in text_cache]
    if uncached:
        with torch.inference_mode():
            tokens = tok(uncached).to(device)
            v = model.encode_text(tokens)
            v = (v / v.norm(dim=-1, keepdim=True)).float().cpu().numpy()
            for t, vec in zip(uncached, v):
                text_cache[t] = vec
    return np.stack([text_cache[t] for t in texts])

def encode_pil_image(pil_img):
    with torch.inference_mode():
        t = pre(pil_img).unsqueeze(0).to(device)
        v = model.encode_image(t)[0]
        return (v / v.norm()).float().cpu().numpy()

def encode_pil_images_batch(pil_imgs):
    with torch.inference_mode():
        tensors = torch.stack([pre(img) for img in pil_imgs]).to(device)
        v = model.encode_image(tensors)
        return (v / v.norm(dim=-1, keepdim=True)).float().cpu().numpy()

def decode_image_input(s: str) -> Image.Image:
    s = s.strip()
    if s.startswith(("http://", "https://")):
        import urllib.request
        req = urllib.request.Request(s, headers={"User-Agent": "ImageJev-Colab/0.1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    if s.startswith("data:image/") and ";base64," in s:
        s = s.split(";base64,")[1]
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")

# ------------------------------------------------------------------------------
# FASTAPI APPLICATION
# ------------------------------------------------------------------------------
app = FastAPI(
    title="ImageJev GPU Cloud Server",
    version="0.1.0",
    description="High-throughput GPU-accelerated closed-schema decision engine."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DecideReq(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    image: str
    decision: str
    schema_: dict = Field(default_factory=dict, alias="schema")

class DecideBatchReq(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    images: List[str]
    decision: str
    schema_: dict = Field(default_factory=dict, alias="schema")

class SystemOneReq(BaseModel):
    image: str
    questions: Dict[str, dict]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "imagejev",
        "version": "0.1.0",
        "device": gpu_name,
        "cuda": torch.cuda.is_available()
    }

@app.post("/decide")
def decide_endpoint(req: DecideReq):
    t0 = time.perf_counter()
    img = decode_image_input(req.image)
    img_v = encode_pil_image(img)
    options = req.schema_.get("options", {})
    opt_keys = list(options.keys())
    text_v = get_text_vecs([f"{k}: a photo of {v}" for k, v in options.items()])
    logits = (img_v @ text_v.T) * logit_scale
    e = np.exp(logits - np.max(logits))
    probs = e / np.sum(e)
    top_idx = int(np.argmax(probs))
    sorted_p = np.sort(probs)[::-1]
    margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_probs := sorted_p) > 1 else float(probs[0])
    gpu_ms = (time.perf_counter() - t0) * 1000
    return {
        "decision": req.decision,
        "selected": opt_keys[top_idx],
        "confidence": margin,
        "probabilities": {k: float(p) for k, p in zip(opt_keys, probs)},
        "server_gpu_ms": round(gpu_ms, 2)
    }

@app.post("/decide_upload")
async def decide_upload_endpoint(
    file: UploadFile = File(...),
    decision: str = Form("What is in this image?"),
    schema: str = Form('{"options": {"yes": "yes", "no": "no"}}')
):
    t0 = time.perf_counter()
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    schema_dict = json.loads(schema)
    img_v = encode_pil_image(img)
    options = schema_dict.get("options", {})
    opt_keys = list(options.keys())
    text_v = get_text_vecs([f"{k}: a photo of {v}" for k, v in options.items()])
    logits = (img_v @ text_v.T) * logit_scale
    e = np.exp(logits - np.max(logits))
    probs = e / np.sum(e)
    top_idx = int(np.argmax(probs))
    sorted_p = np.sort(probs)[::-1]
    margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else float(probs[0])
    gpu_ms = (time.perf_counter() - t0) * 1000
    return {
        "filename": file.filename,
        "decision": decision,
        "selected": opt_keys[top_idx],
        "confidence": margin,
        "probabilities": {k: float(p) for k, p in zip(opt_keys, probs)},
        "server_gpu_ms": round(gpu_ms, 2)
    }

@app.post("/v1/systemone")
def systemone_endpoint(req: SystemOneReq):
    t0 = time.perf_counter()
    img = decode_image_input(req.image)
    img_v = encode_pil_image(img)
    answers = {}
    for q_id, q_data in req.questions.items():
        opts = q_data.get("schema", {}).get("options", {})
        opt_keys = list(opts.keys())
        text_v = get_text_vecs([f"{k}: a photo of {v}" for k, v in opts.items()])
        logits = (img_v @ text_v.T) * logit_scale
        e = np.exp(logits - np.max(logits))
        probs = e / np.sum(e)
        top_idx = int(np.argmax(probs))
        answers[q_id] = {
            "selected": opt_keys[top_idx],
            "confidence": float(probs[top_idx]),
            "probabilities": {k: float(p) for k, p in zip(opt_keys, probs)}
        }
    gpu_ms = (time.perf_counter() - t0) * 1000
    return {"answers": answers, "server_gpu_ms": round(gpu_ms, 2)}

@app.post("/decide_batch")
def decide_batch_endpoint(req: DecideBatchReq):
    t0 = time.perf_counter()
    imgs = [decode_image_input(s) for s in req.images]
    img_vecs = encode_pil_images_batch(imgs)
    options = req.schema_.get("options", {})
    opt_keys = list(options.keys())
    text_v = get_text_vecs([f"{k}: a photo of {v}" for k, v in options.items()])
    logits = (img_vecs @ text_v.T) * logit_scale
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = e / np.sum(e, axis=-1, keepdims=True)
    gpu_ms = (time.perf_counter() - t0) * 1000
    return {
        "results": [{"selected": opt_keys[int(np.argmax(r))], "confidence": float(np.max(r))} for r in probs],
        "batch_size": len(imgs),
        "total_server_gpu_ms": round(gpu_ms, 2),
        "per_image_ms": round(gpu_ms / len(imgs), 2)
    }

# ------------------------------------------------------------------------------
# START UVICORN + CLOUDFLARE TUNNEL
# ------------------------------------------------------------------------------
def run_uvicorn():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

threading.Thread(target=run_uvicorn, daemon=True).start()
time.sleep(2)
print("✅ Uvicorn running on http://127.0.0.1:8000")

# Download Cloudflared
if not os.path.exists("cloudflared"):
    print("Downloading Cloudflare Tunnel binary...")
    subprocess.run(["wget", "-q", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", "cloudflared"])
    subprocess.run(["chmod", "+x", "cloudflared"])

print("\n" + "=" * 70)
print("🌐 ESTABLISHING PUBLIC HTTPS TUNNEL...")
print("=" * 70)
tunnel_proc = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Parse tunnel URL from stderr
public_url = None
for line in iter(tunnel_proc.stderr.readline, ""):
    if "trycloudflare.com" in line:
        for part in line.split():
            if "trycloudflare.com" in part and part.startswith("http"):
                public_url = part.strip()
                break
        if public_url:
            break

print("=" * 70)
print(f"🚀 SERVER IS LIVE!")
print(f"👉 PUBLIC SERVER URL:  {public_url}")
print(f"👉 SWAGGER DOCS URL:   {public_url}/docs")
print("=" * 70)
print("\nCopy this Public Server URL and paste it into Colab #2 (Client) or your laptop!\n")
