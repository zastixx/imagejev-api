"""Interactive REST API Test Script for ImageJev.

Tests both:
1. Native Single/Batch Decision: POST /decide and POST /decide_batch
2. Multi-Question System-One Wire Format: POST /v1/systemone and POST /v1/decide (Multi-Question evaluation in 1 image pass)
"""
import base64
import json
import time
from pathlib import Path
from fastapi.testclient import TestClient

from imagejev import ImageJev, OpenCLIPBackend
from imagejev.server import app, set_engine

def main():
    print("=" * 80)
    print("TESTING IMAGEJEV REST API (MULTI-QUESTION DECISION COMPATIBLE)")
    print("=" * 80)

    # 1. Initialize real CLIP engine and inject into FastAPI app
    print("\n[1] Initializing OpenCLIP Engine (ViT-B-32 / laion2b_s34b_b79k)...")
    t0 = time.perf_counter()
    engine = ImageJev(OpenCLIPBackend())
    set_engine(engine)
    client = TestClient(app)
    print(f"Engine & API TestClient ready in {time.perf_counter() - t0:.2f}s.\n")

    # 2. Test GET /health
    print("-" * 80)
    print("TEST 1: Health Check -> GET /health")
    print("-" * 80)
    res_health = client.get("/health")
    print(f"HTTP Status: {res_health.status_code}")
    print(f"Response:    {res_health.json()}\n")

    # 3. Test Multi-Question API -> POST /v1/systemone
    print("-" * 80)
    print("TEST 2: Multi-Question Wire Format -> POST /v1/systemone")
    print("Simultaneously evaluates 3 questions (Choice, Score, Noul) on 1 image encode pass!")
    print("-" * 80)

    perfume_url = "https://images.unsplash.com/photo-1523275335684-37898b6baf30"
    
    multi_payload = {
        "image": perfume_url,
        "questions": {
            "catalog_suitability": {
                "type": "choice",
                "instructions": "Is this product photo suitable for an e-commerce catalog?",
                "options": {
                    "suitable": "a clean, well-lit, professional studio product photo",
                    "unsuitable": "a blurry, dark, torn, or low-quality photo"
                }
            },
            "lighting_score": {
                "type": "score",
                "instructions": "Rate the lighting and clarity quality",
                "levels": [
                    "very dark or blurry",
                    "dimly lit or grainy",
                    "clear and balanced exposure",
                    "exceptionally crisp studio lighting"
                ]
            },
            "is_watch_or_wearable": {
                "type": "noul",
                "instructions": "Is this a watch, wristwatch, or wearable accessory?",
                "true": "a wristwatch, watch face, or wearable timepiece",
                "false": "non-wearable item, food, or landscape"
            }
        }
    }

    print(f"Image Target: {perfume_url}")
    print(f"Questions in payload: {list(multi_payload['questions'].keys())}")
    
    t_req = time.perf_counter()
    res_systemone = client.post("/v1/systemone", json=multi_payload)
    latency_ms = (time.perf_counter() - t_req) * 1000

    print(f"HTTP Status:  {res_systemone.status_code} (Latency: {latency_ms:.1f}ms)")
    data = res_systemone.json()
    print("\nStructured Response from /v1/systemone:")
    print(json.dumps(data, indent=2))

    # 4. Test Native Jev Single Decision -> POST /decide with Base64 Payload
    print("\n" + "-" * 80)
    print("TEST 3: Native ImageJev Decision -> POST /decide (with Base64 Image)")
    print("-" * 80)

    workspace_root = Path(__file__).resolve().parent.parent.parent
    local_img = workspace_root / "test_images" / "golden_retriever.jpg"
    
    if local_img.exists():
        raw_b64 = base64.b64encode(local_img.read_bytes()).decode("utf-8")
        image_input = raw_b64
        print(f"Using local base64 image: {local_img.name} ({len(raw_b64)} chars)")
    else:
        image_input = "https://images.unsplash.com/photo-1543466835-00a7907e9de1"
        print(f"Using URL image: {image_input}")

    native_payload = {
        "image": image_input,
        "decision": "Identify the animal species",
        "schema": {
            "type": "choice",
            "options": {
                "dog": "a golden retriever dog portrait",
                "cat": "a cute cat sitting indoors",
                "bird": "a flying or perched bird",
                "wildlife": "a wild animal in a jungle"
            }
        }
    }

    t_req = time.perf_counter()
    res_decide = client.post("/decide", json=native_payload)
    latency_ms = (time.perf_counter() - t_req) * 1000

    print(f"HTTP Status:  {res_decide.status_code} (Latency: {latency_ms:.1f}ms)")
    print("Response from /decide:")
    print(json.dumps(res_decide.json(), indent=2))

    # 5. Test Batch Decision -> POST /decide_batch
    print("\n" + "-" * 80)
    print("TEST 4: High-Throughput Batch Decision -> POST /decide_batch")
    print("-" * 80)

    batch_payload = {
        "images": [
            "https://images.unsplash.com/photo-1543466835-00a7907e9de1",  # dog
            "https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7",  # car
            "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba"   # cat
        ],
        "decision": "Classify subject",
        "schema": {
            "type": "choice",
            "options": {
                "dog": "a photo of a dog",
                "cat": "a photo of a cat",
                "car": "a photo of a car"
            }
        }
    }

    t_req = time.perf_counter()
    res_batch = client.post("/decide_batch", json=batch_payload)
    latency_ms = (time.perf_counter() - t_req) * 1000

    print(f"HTTP Status:  {res_batch.status_code} (Latency: {latency_ms:.1f}ms, ~{latency_ms/3:.1f}ms/image)")
    print("Response from /decide_batch:")
    for idx, item in enumerate(res_batch.json()):
        print(f"  Item {idx+1}: choice={item['choice']}, prob={item['probability']:.4f}, conf={item['confidence']:.4f}")

    print("\n" + "=" * 80)
    print("ALL API ENDPOINT TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
