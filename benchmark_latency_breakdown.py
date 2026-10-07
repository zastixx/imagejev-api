"""Micro-benchmark: Dissecting latency breakdown for ImageJev.

Measures:
1. Network download overhead vs Local bytes/PIL.
2. Cold vs Warm inference latency (Cached text vectors).
3. Pure vision backbone forward pass vs closed-schema math.
4. Batch throughput (1 vs 4 vs 8 images).
"""
import time
from pathlib import Path
from PIL import Image
import numpy as np

from imagejev import ImageJev, OpenCLIPBackend

def run_benchmark():
    print("=" * 80)
    print("IMAGEJEV LATENCY BREAKDOWN & PERFORMANCE PROFILING")
    print("=" * 80)

    print("\n[1] Initializing OpenCLIP Backend...")
    t0 = time.perf_counter()
    backend = OpenCLIPBackend()
    jev = ImageJev(backend=backend)
    print(f"Backend device: {backend.device} (Torch version: {backend.torch.__version__})")

    # Load local test image into memory
    workspace_root = Path(__file__).resolve().parent.parent
    img_path = workspace_root / "test_images" / "golden_retriever.jpg"
    if not img_path.exists():
        img = Image.new("RGB", (224, 224), color=(255, 128, 0))
    else:
        img = Image.open(img_path).convert("RGB")

    schema = {
        "type": "choice",
        "options": {
            "dog": "a golden retriever dog portrait",
            "cat": "a cute cat sitting indoors",
            "car": "a red sports car on road",
            "nature": "a landscape of mountains and trees"
        }
    }

    # -------------------------------------------------------------------------
    # 1. Warm-up and Text Cache Populating
    # -------------------------------------------------------------------------
    print("\n[2] Warming up model & caching prompt vectors...")
    _ = jev.decide(img, "Identify object", schema)

    # -------------------------------------------------------------------------
    # 2. Pure Inference Latency on Local In-Memory Image (Warm)
    # -------------------------------------------------------------------------
    runs = 20
    latencies = []
    for _ in range(runs):
        t_start = time.perf_counter()
        _ = jev.decide(img, "Identify object", schema)
        latencies.append((time.perf_counter() - t_start) * 1000)

    avg_lat = np.mean(latencies)
    min_lat = np.min(latencies)
    p95_lat = np.percentile(latencies, 95)

    print(f"\n--- In-Memory Decision Latency ({runs} iterations) ---")
    print(f"  Average:  {avg_lat:.2f} ms")
    print(f"  Minimum:  {min_lat:.2f} ms")
    print(f"  P95:      {p95_lat:.2f} ms")

    # -------------------------------------------------------------------------
    # 3. Component Breakdown: Vision Encode vs Decision Math
    # -------------------------------------------------------------------------
    t_enc_start = time.perf_counter()
    for _ in range(runs):
        _ = backend.encode_image(img)
    avg_enc = ((time.perf_counter() - t_enc_start) / runs) * 1000

    img_vec = backend.encode_image(img)
    text_vecs = jev._embed_batch(list(schema["options"].values()))
    
    t_math_start = time.perf_counter()
    for _ in range(1000):
        logits = backend.logit_scale * (text_vecs @ img_vec)
        e = np.exp(logits - np.max(logits))
        _ = e / np.sum(e)
    avg_math_us = ((time.perf_counter() - t_math_start) / 1000) * 1_000_000  # microseconds!

    print(f"\n--- Component Breakdown ---")
    print(f"  1. Vision Transformer Forward Pass: {avg_enc:.2f} ms ({avg_enc/avg_lat*100:.1f}%)")
    print(f"  2. Closed-Schema Decision Math:      {avg_math_us:.1f} µs ({avg_math_us/1000/avg_lat*100:.2f}%)")

    # -------------------------------------------------------------------------
    # 4. Batch Throughput
    # -------------------------------------------------------------------------
    batch_sizes = [1, 4, 8]
    print(f"\n--- Batched Inference Throughput ---")
    for bs in batch_sizes:
        batch_imgs = [img] * bs
        t_b_start = time.perf_counter()
        _ = jev.decide_batch(batch_imgs, "Identify object", schema)
        b_time_ms = (time.perf_counter() - t_b_start) * 1000
        per_img = b_time_ms / bs
        print(f"  Batch Size {bs:2d}: {b_time_ms:6.1f} ms total -> {per_img:5.1f} ms / image")

    # -------------------------------------------------------------------------
    # 5. Network URL Download vs In-Memory Comparison
    # -------------------------------------------------------------------------
    import urllib.request
    url = "https://images.unsplash.com/photo-1523275335684-37898b6baf30"
    t_net_start = time.perf_counter()
    req = urllib.request.Request(url, headers={"User-Agent": "ImageJev-SDK"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            _ = resp.read()
        net_ms = (time.perf_counter() - t_net_start) * 1000

        print(f"\n--- Network vs Local Overhead ---")
        print(f"  Internet Download (Unsplash HTTP):  {net_ms:.1f} ms")
        print(f"  Model Inference (In-Memory CPU):     {avg_lat:.1f} ms")
        print(f"  -> Network download accounted for {net_ms/(net_ms+avg_lat)*100:.1f}% of total API response time!")
    except Exception as e:
        print(f"Network benchmark skipped: {e}")

if __name__ == "__main__":
    run_benchmark()
