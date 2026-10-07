"""
================================================================================
IMAGEJEV: GOOGLE COLAB FASTAPI REST API ENDPOINT BENCHMARK
================================================================================
Self-contained script to test all ImageJev REST API endpoints on Tesla T4 GPU
inside Google Colab.

Endpoints tested:
1. GET  /health          -> Service status & health check
2. POST /decide          -> Single image (Base64 / URL) structured decision
3. POST /decide_upload   -> Multipart file upload decision
4. POST /decide_batch    -> High-throughput batch decisions
5. POST /v1/systemone    -> Multi-question concurrent evaluation
================================================================================
"""

import io
import time
import base64
from PIL import Image
import torch
import numpy as np

# Dependencies check
import open_clip
from fastapi import FastAPI, File, Form, UploadFile, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from typing import Any, Dict, List

# ------------------------------------------------------------------------------
# 1. INITIALIZE FASTAPI + GPU BACKEND
# ------------------------------------------------------------------------------
print("=" * 70)
print("INITIALIZING IMAGEJEV FASTAPI SERVICE ON TESLA T4 GPU...")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

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

# ------------------------------------------------------------------------------
# 2. FASTAPI APPLICATION SETUP
# ------------------------------------------------------------------------------
app = FastAPI(title="ImageJev API (Colab T4)", version="0.1.0")

class DecideRequest(BaseModel):
    image: str
    decision: str
    schema: dict

class DecideBatchRequest(BaseModel):
    images: List[str]
    decision: str
    schema: dict

class SystemOneRequest(BaseModel):
    image: str
    questions: Dict[str, dict]

def decode_image_base64_or_bytes(raw: str) -> Image.Image:
    if raw.startswith("data:image/") and ";base64," in raw:
        raw = raw.split(";base64,")[1]
    data = base64.b64decode(raw)
    return Image.open(io.BytesIO(data)).convert("RGB")

@app.get("/health")
def health():
    return {"status": "ok", "service": "imagejev", "version": "0.1.0", "device": device}

@app.post("/decide")
def decide(req: DecideRequest):
    img = decode_image_base64_or_bytes(req.image)
    img_v = encode_pil_image(img)
    options = req.schema.get("options", {})
    opt_keys = list(options.keys())
    opt_texts = [f"{k}: a photo of {v}" for k, v in options.items()]
    text_v = get_text_vecs(opt_texts)
    logits = (img_v @ text_v.T) * logit_scale
    e = np.exp(logits - np.max(logits))
    probs = e / np.sum(e)
    top_idx = int(np.argmax(probs))
    sorted_probs = np.sort(probs)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(probs[0])
    return {
        "decision": req.decision,
        "selected": opt_keys[top_idx],
        "confidence": margin,
        "probabilities": {k: float(p) for k, p in zip(opt_keys, probs)}
    }

@app.post("/decide_upload")
async def decide_upload(file: UploadFile = File(...), decision: str = Form(...), schema: str = Form(...)):
    import json
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    schema_dict = json.loads(schema)
    img_v = encode_pil_image(img)
    options = schema_dict.get("options", {})
    opt_keys = list(options.keys())
    opt_texts = [f"{k}: a photo of {v}" for k, v in options.items()]
    text_v = get_text_vecs(opt_texts)
    logits = (img_v @ text_v.T) * logit_scale
    e = np.exp(logits - np.max(logits))
    probs = e / np.sum(e)
    top_idx = int(np.argmax(probs))
    return {
        "decision": decision,
        "selected": opt_keys[top_idx],
        "confidence": float(probs[top_idx]),
        "probabilities": {k: float(p) for k, p in zip(opt_keys, probs)}
    }

@app.post("/decide_batch")
def decide_batch(req: DecideBatchRequest):
    imgs = [decode_image_base64_or_bytes(s) for s in req.images]
    img_vecs = encode_pil_images_batch(imgs)
    options = req.schema.get("options", {})
    opt_keys = list(options.keys())
    opt_texts = [f"{k}: a photo of {v}" for k, v in options.items()]
    text_v = get_text_vecs(opt_texts)
    logits = (img_vecs @ text_v.T) * logit_scale
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = e / np.sum(e, axis=-1, keepdims=True)
    res = []
    for row in probs:
        top_idx = int(np.argmax(row))
        res.append({"selected": opt_keys[top_idx], "probability": float(row[top_idx])})
    return res

@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    img = decode_image_base64_or_bytes(req.image)
    img_v = encode_pil_image(img)
    answers = {}
    for q_id, q_data in req.questions.items():
        opts = q_data.get("schema", {}).get("options", {})
        opt_keys = list(opts.keys())
        opt_texts = [f"{k}: a photo of {v}" for k, v in opts.items()]
        text_v = get_text_vecs(opt_texts)
        logits = (img_v @ text_v.T) * logit_scale
        e = np.exp(logits - np.max(logits))
        probs = e / np.sum(e)
        top_idx = int(np.argmax(probs))
        answers[q_id] = {
            "selected": opt_keys[top_idx],
            "confidence": float(probs[top_idx]),
            "probabilities": {k: float(p) for k, p in zip(opt_keys, probs)}
        }
    return {"answers": answers}

# ------------------------------------------------------------------------------
# 3. TEST CLIENT BENCHMARKS
# ------------------------------------------------------------------------------
client = TestClient(app)

# Helper: Create test image base64
sample_img = Image.new("RGB", (224, 224), color=(220, 100, 50))
buf = io.BytesIO()
sample_img.save(buf, format="JPEG")
sample_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

print("\n" + "=" * 70)
print("TEST 1: GET /health")
print("=" * 70)
t0 = time.perf_counter()
resp = client.get("/health")
t_health = (time.perf_counter() - t0) * 1000
print(f"Status:   {resp.status_code}")
print(f"Payload:  {resp.json()}")
print(f"Latency:  {t_health:.2f} ms")

print("\n" + "=" * 70)
print("TEST 2: POST /decide (Single Image Structured Decision)")
print("=" * 70)
payload_decide = {
    "image": sample_b64,
    "decision": "What is the primary subject in this image?",
    "schema": {
        "type": "choice",
        "options": {
            "shoes": "shoes sneakers footwear",
            "electronics": "smartphone laptop gadgets",
            "food": "fresh food pizza meal beverage",
            "vehicle": "automobile car truck motorcycle"
        }
    }
}
# Warmup
client.post("/decide", json=payload_decide)

# Benchmark 20 iterations
times = []
for _ in range(20):
    t0 = time.perf_counter()
    resp = client.post("/decide", json=payload_decide)
    times.append((time.perf_counter() - t0) * 1000)

print(f"Status:          {resp.status_code}")
print(f"Selected:        >>> {resp.json()['selected'].upper()} <<< (Confidence: {resp.json()['confidence']*100:.1f}%)")
print(f"P50 API Latency: {np.percentile(times, 50):.2f} ms")
print(f"Mean HTTP Roundtrip: {np.mean(times):.2f} ms")

print("\n" + "=" * 70)
print("TEST 3: POST /decide_upload (Multipart File Upload)")
print("=" * 70)
import json
buf.seek(0)
files_payload = {"file": ("test.jpg", buf.getvalue(), "image/jpeg")}
data_payload = {
    "decision": "Is this product image compliant?",
    "schema": json.dumps({
        "type": "choice",
        "options": {
            "approved": "clear studio product photo compliant",
            "rejected": "blurry dark poor resolution unacceptable"
        }
    })
}
t0 = time.perf_counter()
resp = client.post("/decide_upload", files=files_payload, data=data_payload)
t_upload = (time.perf_counter() - t0) * 1000
print(f"Status:          {resp.status_code}")
print(f"Selected:        >>> {resp.json()['selected'].upper()} <<<")
print(f"Upload Latency:  {t_upload:.2f} ms")

print("\n" + "=" * 70)
print("TEST 4: POST /v1/systemone (Multi-Question Concurrent Engine)")
print("=" * 70)
payload_sys1 = {
    "image": sample_b64,
    "questions": {
        "category": {
            "decision": "Category",
            "schema": {"options": {"food": "food dish meal", "shoes": "shoes sneakers", "car": "vehicle car"}}
        },
        "moderation": {
            "decision": "Moderation",
            "schema": {"options": {"safe": "safe family-friendly", "unsafe": "inappropriate content"}}
        },
        "quality": {
            "decision": "Quality",
            "schema": {"options": {"high_quality": "sharp clear bright lighting", "low_quality": "blurry dark"}}
        }
    }
}
times = []
for _ in range(20):
    t0 = time.perf_counter()
    resp = client.post("/v1/systemone", json=payload_sys1)
    times.append((time.perf_counter() - t0) * 1000)

print(f"Status:          {resp.status_code}")
for q_id, q_res in resp.json()["answers"].items():
    print(f"  • {q_id:<12} -> {q_res['selected'].upper():<15} (Confidence: {q_res['confidence']*100:.1f}%)")
print(f"P50 Multi-Q API Latency: {np.percentile(times, 50):.2f} ms (evaluates all 3 questions on 1 pass)")

print("\n" + "=" * 70)
print("TEST 5: POST /decide_batch (Batch-16 High Throughput)")
print("=" * 70)
payload_batch = {
    "images": [sample_b64] * 16,
    "decision": "Classify batch items",
    "schema": payload_decide["schema"]
}
t0 = time.perf_counter()
resp = client.post("/decide_batch", json=payload_batch)
t_batch = (time.perf_counter() - t0) * 1000
print(f"Status:          {resp.status_code}")
print(f"Batch Size:      16 images")
print(f"Total API Time:  {t_batch:.2f} ms ({t_batch / 16:.2f} ms / image)")
print(f"Throughput:      {(16 / (t_batch / 1000)):.1f} images / second over HTTP REST")
print("=" * 70)
