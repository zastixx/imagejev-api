"""
================================================================================
IMAGEJEV: COLAB NOTEBOOK #2 - CLIENT / USER / PUBLIC TESTER
================================================================================
Run this in a second Google Colab notebook (or on your laptop/terminal).

Features:
- Connects to the Colab #1 GPU Server via its public HTTPS URL
- Uploads images from your computer or tests remote image URLs
- Calls all structured decision endpoints over the internet
- Displays real-time Network Roundtrip vs Server GPU execution latency
================================================================================
"""

# [Cell 1]: Install client dependencies
# !pip install -q requests pillow matplotlib

import io
import time
import json
import base64
import requests
import matplotlib.pyplot as plt
from PIL import Image
from google.colab import files

# ------------------------------------------------------------------------------
# 1. SET YOUR COLAB #1 PUBLIC SERVER URL HERE
# ------------------------------------------------------------------------------
SERVER_URL = input("Enter your Colab #1 Server URL (e.g., https://xxxx.trycloudflare.com): ").strip()
if SERVER_URL.endswith("/"):
    SERVER_URL = SERVER_URL[:-1]

# ------------------------------------------------------------------------------
# 2. CHECK SERVER HEALTH
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"Connecting to ImageJev GPU Server at: {SERVER_URL}...")
print("=" * 70)

try:
    t0 = time.perf_counter()
    health_resp = requests.get(f"{SERVER_URL}/health", timeout=10)
    net_ms = (time.perf_counter() - t0) * 1000
    if health_resp.status_code == 200:
        data = health_resp.json()
        print(f"✅ Connection successful! (Ping: {net_ms:.1f} ms)")
        print(f"   Server Device:  {data.get('device')}")
        print(f"   Service:        {data.get('service')} v{data.get('version')}")
    else:
        print(f"❌ Server returned status {health_resp.status_code}: {health_resp.text}")
except Exception as e:
    print(f"❌ Failed to connect to server: {e}")
    raise SystemExit

# ------------------------------------------------------------------------------
# 3. UPLOAD AN IMAGE OR USE A TEST PHOTO
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SELECT AN IMAGE TO SEND TO THE GPU SERVER")
print("=" * 70)
print("1. Upload a file from your computer")
print("2. Use a sample online photo (Red sneakers)")
choice = input("Choose option [1 or 2, default 1]: ").strip() or "1"

if choice == "1":
    print("\nClick 'Choose Files' to upload an image from your laptop/phone:")
    uploaded = files.upload()
    if not uploaded:
        print("No file chosen. Exiting.")
        raise SystemExit
    filename = list(uploaded.keys())[0]
    img_bytes = uploaded[filename]
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
else:
    sample_url = "https://images.unsplash.com/photo-1542291026-7eec264c27ff"
    print(f"\nDownloading test image from {sample_url}...")
    resp = requests.get(sample_url, timeout=10)
    img_bytes = resp.content
    filename = "sample_sneakers.jpg"
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

# Display Image
plt.figure(figsize=(4, 4))
plt.imshow(pil_image)
plt.axis("off")
plt.title(f"Client Image: {filename}", fontsize=11)
plt.show()

# ------------------------------------------------------------------------------
# 4. TEST MULTIPART FILE UPLOAD (POST /decide_upload)
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 1: MULTIPART FILE UPLOAD -> POST /decide_upload")
print("=" * 70)

decision = "What is the primary subject in this photo?"
schema = {
    "options": {
        "shoes": "shoes footwear sneakers boots",
        "pet": "dog cat animal pet",
        "vehicle": "car truck motorcycle vehicle",
        "food": "food dish meal beverage snack",
        "nature": "nature landscape mountains tree beach",
        "person": "human face portrait person"
    }
}

t0 = time.perf_counter()
resp = requests.post(
    f"{SERVER_URL}/decide_upload",
    files={"file": (filename, img_bytes, "image/jpeg")},
    data={"decision": decision, "schema": json.dumps(schema)},
    timeout=15
)
total_network_ms = (time.perf_counter() - t0) * 1000

if resp.status_code == 200:
    res = resp.json()
    print(f"Decision:           '{res['decision']}'")
    print(f"Selected:           >>> {res['selected'].upper()} <<<")
    print(f"Confidence Margin:  {res['confidence'] * 100:.1f}%")
    print(f"Server GPU Time:    ⚡ {res.get('server_gpu_ms', 'N/A')} ms")
    print(f"Total Client Ping:  🌐 {total_network_ms:.1f} ms (including upload over internet)")
    print("\nProbabilities:")
    for k, p in res["probabilities"].items():
        bar = "#" * int(p * 25)
        print(f"  • {k:<15} {p * 100:5.1f}%  [{bar:<25}]")
else:
    print(f"Error {resp.status_code}: {resp.text}")

# ------------------------------------------------------------------------------
# 5. TEST MULTI-QUESTION SYSTEM-ONE ENDPOINT (POST /v1/systemone)
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 2: MULTI-QUESTION CONCURRENT DECISIONS -> POST /v1/systemone")
print("Simultaneously evaluating Moderation, Quality Score, and Classification:")
print("=" * 70)

b64_str = base64.b64encode(img_bytes).decode("utf-8")
multi_q_payload = {
    "image": b64_str,
    "questions": {
        "moderation": {
            "schema": {"options": {"safe": "safe family friendly listing", "unsafe": "inappropriate objectionable content"}}
        },
        "quality": {
            "schema": {"options": {"high_quality": "clear crisp sharp studio photo", "low_quality": "blurry dark degraded"}}
        },
        "subject": {
            "schema": schema
        }
    }
}

t0 = time.perf_counter()
resp = requests.post(f"{SERVER_URL}/v1/systemone", json=multi_q_payload, timeout=15)
multi_net_ms = (time.perf_counter() - t0) * 1000

if resp.status_code == 200:
    res = resp.json()
    print(f"Server GPU Time:    ⚡ {res.get('server_gpu_ms', 'N/A')} ms (for all 3 questions!)")
    print(f"Total Client Ping:  🌐 {multi_net_ms:.1f} ms")
    print("\nStructured Answers:")
    for q_id, q_data in res["answers"].items():
        print(f"  • [{q_id.upper()}] Selected: >>> {q_data['selected'].upper()} <<<  (Confidence: {q_data['confidence']*100:.1f}%)")
else:
    print(f"Error {resp.status_code}: {resp.text}")

print("\n" + "=" * 70)
print("✅ MULTI-COLAB DISTRIBUTED TEST COMPLETE!")
print("=" * 70)
