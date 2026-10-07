"""
================================================================================
IMAGEJEV: GOOGLE COLAB NVIDIA T4 GPU BENCHMARK SUITE
================================================================================
Comprehensive benchmark to measure latency and throughput of ImageJev on 
Google Colab (Tesla T4 GPU).

To run on Google Colab:
1. Open https://colab.research.google.com
2. Set Runtime -> Change runtime type -> T4 GPU (Hardware Accelerator: GPU)
3. In a Colab cell, install dependencies:
   !pip install -q open-clip-torch torchvision pillow pydantic

4. Run this script!
================================================================================
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image
import torch

# ------------------------------------------------------------------------------
# 1. GPU ENVIRONMENT DETECTION
# ------------------------------------------------------------------------------
def inspect_gpu():
    print("=" * 80)
    print("IMAGEJEV: HARDWARE & ACCELERATION ENVIRONMENT")
    print("=" * 80)
    print(f"PyTorch Version:       {torch.__version__}")
    print(f"CUDA Available:        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_ver = torch.version.cuda
        cudnn_ver = torch.backends.cudnn.version()
        print(f"GPU Device:            {gpu_name}")
        print(f"Total VRAM:            {vram_gb:.2f} GB")
        print(f"CUDA Version:          {cuda_ver}")
        print(f"cuDNN Version:         {cudnn_ver}")
        print(f"Tensor Cores Active:   {'Yes (Tesla T4 Architecture)' if 'T4' in gpu_name else 'Supported'}")
    else:
        print("WARNING: CUDA is not available! Running on CPU fallback.")
    print("=" * 80)
    print()


# ------------------------------------------------------------------------------
# 2. IMPORT OR DEFINE IMAGEJEV SDK
# ------------------------------------------------------------------------------
try:
    from imagejev import ImageJev
    from imagejev.backends import OpenCLIPBackend
except ImportError:
    import open_clip

    class OpenCLIPBackend:
        def __init__(self, name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=None):
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model, _, self.pre = open_clip.create_model_and_transforms(
                name, pretrained=pretrained, device=self.device
            )
            self.model.eval()
            self.tok = open_clip.get_tokenizer(name)
            self.logit_scale = float(self.model.logit_scale.exp().detach().cpu().item())

        def encode_image(self, pil_img):
            with torch.inference_mode():
                tensor = self.pre(pil_img).unsqueeze(0).to(self.device)
                v = self.model.encode_image(tensor)[0]
                v = (v / v.norm()).float().cpu().numpy()
                return v

        def encode_images(self, pil_images):
            with torch.inference_mode():
                tensors = torch.stack([self.pre(img) for img in pil_images]).to(self.device)
                v = self.model.encode_image(tensors)
                v = (v / v.norm(dim=-1, keepdim=True)).float().cpu().numpy()
                return v

        def encode_text(self, texts):
            with torch.inference_mode():
                tokens = self.tok(list(texts)).to(self.device)
                v = self.model.encode_text(tokens)
                v = (v / v.norm(dim=-1, keepdim=True)).float().cpu().numpy()
                return v

    class ImageJev:
        def __init__(self, backend=None, model="ViT-B-32", pretrained="laion2b_s34b_b79k", device=None):
            self.backend = backend or OpenCLIPBackend(name=model, pretrained=pretrained, device=device)
            self._cache = {}

        def _embed_batch(self, texts):
            uncached = [t for t in texts if t not in self._cache]
            if uncached:
                encoded = self.backend.encode_text(uncached)
                for t, v in zip(uncached, encoded):
                    self._cache[t] = v
            return np.stack([self._cache[t] for t in texts])

        def decide(self, image, decision, schema):
            options = schema.get("options", {})
            opt_keys = list(options.keys())
            opt_texts = [f"{k}: a photo of {v}" for k, v in options.items()]
            img_vec = self.backend.encode_image(image)
            text_vecs = self._embed_batch(opt_texts)
            logits = (img_vec @ text_vecs.T) * self.backend.logit_scale
            e = np.exp(logits - np.max(logits))
            probs = e / np.sum(e)
            top_idx = int(np.argmax(probs))
            sorted_probs = np.sort(probs)[::-1]
            margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(probs[0])
            return {
                "selected": opt_keys[top_idx],
                "confidence": margin,
                "probabilities": {k: float(p) for k, p in zip(opt_keys, probs)}
            }

        def decide_batch(self, images, decision, schema):
            options = schema.get("options", {})
            opt_keys = list(options.keys())
            opt_texts = [f"{k}: a photo of {v}" for k, v in options.items()]
            img_vecs = self.backend.encode_images(images)
            text_vecs = self._embed_batch(opt_texts)
            logits = (img_vecs @ text_vecs.T) * self.backend.logit_scale
            e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = e / np.sum(e, axis=-1, keepdims=True)
            results = []
            for row in probs:
                top_idx = int(np.argmax(row))
                results.append(opt_keys[top_idx])
            return results

        def decide_multi(self, image, questions):
            img_vec = self.backend.encode_image(image)
            results = {}
            for q_id, q_data in questions.items():
                schema = q_data.get("schema", {})
                options = schema.get("options", {})
                opt_keys = list(options.keys())
                opt_texts = [f"{k}: a photo of {v}" for k, v in options.items()]
                text_vecs = self._embed_batch(opt_texts)
                logits = (img_vec @ text_vecs.T) * self.backend.logit_scale
                e = np.exp(logits - np.max(logits))
                probs = e / np.sum(e)
                top_idx = int(np.argmax(probs))
                results[q_id] = opt_keys[top_idx]
            return results


# ------------------------------------------------------------------------------
# 3. BENCHMARK EXECUTION
# ------------------------------------------------------------------------------
def run_benchmark():
    inspect_gpu()

    print("[Step 1/5] Loading OpenCLIP ViT-B-32 Backbone into Memory (Cold Start)...")
    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jev = ImageJev(device=device)
    cold_start_time = (time.perf_counter() - t0) * 1000
    print(f"-> Model Loaded in: {cold_start_time:.2f} ms")
    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated(0) / (1024**2)
        reserved_mb = torch.cuda.memory_reserved(0) / (1024**2)
        print(f"-> GPU VRAM Allocated: {allocated_mb:.1f} MB (Reserved: {reserved_mb:.1f} MB)")
    print()

    # Create synthetic test image (224x224 RGB PIL image)
    test_img = Image.new("RGB", (224, 224), color=(128, 64, 200))
    decision = "What product category is in this image?"
    schema = {
        "type": "choice",
        "options": {
            "electronics": "electronic devices and gadgets",
            "apparel": "clothing, shoes, and wearable apparel",
            "home": "home goods and furniture",
            "beauty": "beauty and cosmetic products",
            "groceries": "food and grocery items",
        }
    }

    # Warm-up iterations
    print("[Step 2/5] Warming up GPU Kernels & cuDNN...")
    for _ in range(10):
        jev.decide(test_img, decision=decision, schema=schema)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("-> Warmup complete (10 cycles).\n")

    # --------------------------------------------------------------------------
    # 4. SINGLE-IMAGE LATENCY DISTRIBUTION (P50, P90, P99)
    # --------------------------------------------------------------------------
    print("[Step 3/5] Profiling Single-Image End-to-End Latency (50 runs)...")
    n_runs = 50
    latencies = []
    embed_latencies = []
    math_latencies = []

    for _ in range(n_runs):
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            mid_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            img_vec = jev.backend.encode_image(test_img)
            mid_event.record()

            # Closed schema decision math with cached text embeddings
            opt_texts = [f"{k}: a photo of {v}" for k, v in schema["options"].items()]
            text_vecs = jev._embed_batch(opt_texts)
            logits = (img_vec @ text_vecs.T) * jev.backend.logit_scale
            e = np.exp(logits - np.max(logits))
            probs = e / np.sum(e)
            end_event.record()

            torch.cuda.synchronize()
            total_ms = start_event.elapsed_time(end_event)
            embed_ms = start_event.elapsed_time(mid_event)
            math_ms = mid_event.elapsed_time(end_event)
        else:
            t_start = time.perf_counter()
            img_vec = jev.backend.encode_image(test_img)
            t_mid = time.perf_counter()
            opt_texts = [f"{k}: a photo of {v}" for k, v in schema["options"].items()]
            text_vecs = jev._embed_batch(opt_texts)
            logits = (img_vec @ text_vecs.T) * jev.backend.logit_scale
            e = np.exp(logits - np.max(logits))
            probs = e / np.sum(e)
            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000
            embed_ms = (t_mid - t_start) * 1000
            math_ms = (t_end - t_mid) * 1000

        latencies.append(total_ms)
        embed_latencies.append(embed_ms)
        math_latencies.append(math_ms)

    latencies = np.array(latencies)
    embed_latencies = np.array(embed_latencies)
    math_latencies = np.array(math_latencies)

    print("-" * 80)
    print(f"SINGLE IMAGE LATENCY BREAKDOWN ({'Tesla T4 GPU' if torch.cuda.is_available() else 'CPU'})")
    print("-" * 80)
    print(f"  Mean Latency:        {np.mean(latencies):6.2f} ms  (Std: {np.std(latencies):.2f} ms)")
    print(f"  P50 (Median):        {np.percentile(latencies, 50):6.2f} ms")
    print(f"  P90:                 {np.percentile(latencies, 90):6.2f} ms")
    print(f"  P95:                 {np.percentile(latencies, 95):6.2f} ms")
    print(f"  P99:                 {np.percentile(latencies, 99):6.2f} ms")
    print(f"  Min / Max:           {np.min(latencies):6.2f} ms / {np.max(latencies):6.2f} ms")
    print()
    print(f"  - ViT Image Encode:  {np.mean(embed_latencies):6.2f} ms  ({(np.mean(embed_latencies)/np.mean(latencies))*100:.1f}% of total)")
    print(f"  - Decision Math:     {np.mean(math_latencies)*1000:6.1f} µs  ({(np.mean(math_latencies)/np.mean(latencies))*100:.2f}% of total)")
    print("-" * 80)
    print()

    # --------------------------------------------------------------------------
    # 5. BATCH THROUGHPUT & LATENCY SCALING
    # --------------------------------------------------------------------------
    print("[Step 4/5] Testing Batch Inference Scaling & Throughput (FPS)...")
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    batch_results = []

    print("-" * 80)
    print(f"{'Batch Size':^12} | {'Total Time (ms)':^18} | {'Per-Image (ms)':^18} | {'Throughput (FPS)':^18}")
    print("-" * 80)

    for b in batch_sizes:
        batch_images = [test_img] * b
        warm_iters = 2
        test_iters = 5

        # Warmup
        for _ in range(warm_iters):
            jev.decide_batch(batch_images, decision=decision, schema=schema)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []
        for _ in range(test_iters):
            if torch.cuda.is_available():
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_end = torch.cuda.Event(enable_timing=True)
                ev_start.record()
                jev.decide_batch(batch_images, decision=decision, schema=schema)
                ev_end.record()
                torch.cuda.synchronize()
                elapsed = ev_start.elapsed_time(ev_end)
            else:
                t0 = time.perf_counter()
                jev.decide_batch(batch_images, decision=decision, schema=schema)
                elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

        avg_batch_ms = np.mean(times)
        per_image_ms = avg_batch_ms / b
        fps = (1000.0 / per_image_ms)
        batch_results.append((b, avg_batch_ms, per_image_ms, fps))
        print(f"{b:^12d} | {avg_batch_ms:15.2f} ms | {per_image_ms:15.2f} ms | {fps:15.1f} img/s")

    print("-" * 80)
    print()

    # --------------------------------------------------------------------------
    # 6. MULTI-QUESTION CONCURRENT EVALUATION ON 1 IMAGE
    # --------------------------------------------------------------------------
    print("[Step 5/5] Multi-Question Structured Decision on Single Visual Embedding...")
    print("Testing 10 simultaneous closed-schema questions on 1 image embedding:")
    
    questions = {
        f"q_{i}": {
            "decision": f"Decision {i}",
            "schema": {
                "type": "choice",
                "options": {
                    "opt_a": f"Option A for question {i}",
                    "opt_b": f"Option B for question {i}",
                    "opt_c": f"Option C for question {i}",
                }
            }
        }
        for i in range(1, 11)
    }

    # Pre-cache text embeddings
    for q_data in questions.values():
        opt_texts = [f"{k}: a photo of {v}" for k, v in q_data["schema"]["options"].items()]
        jev._embed_batch(opt_texts)

    img_vector = jev.backend.encode_image(test_img)

    mq_latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        results = jev.decide_multi(test_img, questions)
        t1 = time.perf_counter()
        mq_latencies.append((t1 - t0) * 1000)

    avg_mq_ms = np.mean(mq_latencies)
    print(f"-> 10 Decisions Evaluated in: {avg_mq_ms:.2f} ms (including single visual encode pass)")
    print()

    # --------------------------------------------------------------------------
    # 7. SUMMARY REPORT
    # --------------------------------------------------------------------------
    print("=" * 80)
    print("IMAGEJEV PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Single Image Median Latency (P50):    {np.percentile(latencies, 50):.2f} ms")
    print(f"Single Image 99th Percentile (P99):   {np.percentile(latencies, 99):.2f} ms")
    print(f"Batch-64 Throughput:                  {batch_results[-1][3]:.1f} Images / Second")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
