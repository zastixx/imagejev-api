# ImageJev

**The Image-Native Structured Decision Engine**

[![PyPI version](https://img.shields.io/pypi/v/imagejev.svg)](https://pypi.org/project/imagejev/)
[![Python versions](https://img.shields.io/pypi/pyversions/imagejev.svg)](https://pypi.org/project/imagejev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

```
Jev:       TEXT   + DECISION SCHEMA -> structured decision
ImageJev:  IMAGE  + DECISION + SCHEMA -> structured decision
```

`imagejev` is a high-throughput, deterministic visual decision engine. It replaces heavy Vision-Language Models (VLMs) with a single frozen CLIP embedding pass and calibrated closed-schema decision math.

- ⚡ **216x lower latency** (~120ms CPU, <15ms GPU)
- 💰 **500–800x lower cost** than proprietary LLM vision APIs
- 🎯 **92.5% accuracy** on CIFAR-10 (statistically indistinguishable from Claude Sonnet 5, McNemar $p=0.23$)
- 🔒 **Zero auto-regressive hallucinations**: 100% schema adherence guarantee

---

## 📦 Installation

```bash
# Core SDK
pip install imagejev

# With FastAPI server support
pip install "imagejev[server]"

# Full development bundle
pip install "imagejev[all]"
```

*Note: For GPU acceleration, ensure PyTorch with CUDA or MPS is installed.*

---

## 🚀 Quickstart

```python
from imagejev import ImageJev

# 1. Initialize engine (loads CLIP ViT-B-32 on first use)
jev = ImageJev()

# 2. Make a structured decision
result = jev.decide(
    image="product.jpg",  # supports local path, URL, PIL.Image, or bytes
    decision="Is this product image suitable for e-commerce listing?",
    schema={
        "type": "choice",
        "options": {
            "yes": "a clean, well-lit, professional in-focus product photo",
            "no": "a blurry, dark, torn, damaged, or cluttered product photo",
        },
    },
)

print(result.choice)        # -> "yes"
print(result.probability)   # -> 0.998
print(result.confidence)    # -> 0.996
print(result.probabilities) # -> {"yes": 0.998, "no": 0.002}
```

---

## 🧩 The Three Decision Primitives

### 1. `choice` Schema (Multi-Class & Quality Filtering)
Evaluates discrete options and returns calibrated probabilities and normalized confidence.

```python
result = jev.decide(
    image="https://images.unsplash.com/photo-1543466835-00a7907e9de1",
    decision="Classify the animal in this photo",
    schema={
        "type": "choice",
        "options": {
            "dog": "a domestic dog or puppy",
            "cat": "a domestic cat or kitten",
            "wildlife": "a wild animal in nature",
        },
    },
)
# Returns: ChoiceResult(choice='dog', probability=0.994, confidence=0.991, probabilities={...})
```

### 2. `score` Schema (Continuous Ratings & Dispersion)
Evaluates ordinal levels (2–10 levels) and computes continuous score expectation, sharpness confidence, and `dispersion` (checks for bimodal disagreement).

```python
result = jev.decide(
    image="photo.jpg",
    decision="Rate lighting and clarity",
    schema={
        "type": "score",
        "levels": [
            "very dark, underexposed, or blurry",
            "dimly lit or slightly grainy",
            "balanced, clear exposure",
            "bright, professional studio lighting",
        ],
    },
)
# Returns: ScoreResult(score=2.85, probability=0.88, confidence=0.84, dispersion=0.01, legend={...})
```

### 3. `noul` Schema (Binary Truth Probability)
Standard binary closed-schema decision:

```python
result = jev.decide(
    image="car.jpg",
    decision="Is there a vehicle on a road?",
    schema={
        "type": "noul",
        "true": "a motorized vehicle or car on a road",
        "false": "nature scenery, animals, or indoor environment",
    },
)
# Returns: NoulResult(probability=0.999)
```

---

## ⚡ High-Throughput Batch Processing

Pass dozens or hundreds of images in a single call to take full advantage of batched tensor matrix operations:

```python
images = [
    "item_1.jpg",
    "item_2.jpg",
    "https://example.com/item_3.jpg",
    "item_4.jpg"
]

results = jev.decide_batch(
    images=images,
    decision="Is this photo compliant with safety guidelines?",
    schema={"type": "choice", "options": {"compliant": "safe", "violation": "unsafe"}}
)

for img, res in zip(images, results):
    print(f"{img}: {res.choice} ({res.confidence:.2f} confidence)")
```

---

## 🌐 Deploy as a REST API Service

Run the production FastAPI service with one command:

```bash
imagejev-server --host 0.0.0.0 --port 8000
```

### HTTP POST `/decide`
```bash
curl -X POST http://localhost:8000/decide \
  -H "X-API-Key: sk-live-yourkey" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "https://images.unsplash.com/photo-1543466835-00a7907e9de1",
    "decision": "Identify the pet",
    "schema": {
      "type": "choice",
      "options": {"dog": "a dog", "cat": "a cat"}
    }
  }'
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### 🛡️ Production Security & Safeguards

ImageJev Server includes built-in safeguards configurable via `.env`:
- **API Key Gate**: Enforced via `X-API-Key` header when `IMAGEJEV_API_KEYS` is defined.
- **Sliding Window Rate Limiter**: 60 req/min for single decisions, 15 req/min for batches.
- **DoS Batch Size Cap**: Up to 50 images per batch request.
- **Payload & SSRF Guards**: 10MB streaming size limit, private IP & link-local network blocking.
- **Concurrency Gate**: Restricts in-flight model inferences to avoid thread thrashing and memory spikes.

---

## 💡 Best Practices for Schema Design

* **Descriptive vs. Bare Options**: For subjective or quality judgments (`yes/no`, `good/bad`), **always provide descriptions** in the schema (`options: {"yes": "clean well-lit photo", "no": "blurry dark photo"}`). Bare options default to `"yes: a photo of a yes"`, which confuses CLIP text encoders.
* **Confidence Gating**: In production pipelines, gate on `result.confidence` rather than `result.choice` alone. Route high-confidence decisions (`> 0.85`) automatically and direct low-confidence cases to human review.

---

## 📊 Benchmark & Accuracy

Validated head-to-head on held-out CIFAR-10 dataset against Anthropic Claude Sonnet 5:

| System | Accuracy | P-Value (vs Sonnet 5) | Latency (CPU) | Cost / 1k Queries |
| :--- | :--- | :--- | :--- | :--- |
| **Claude Sonnet 5** | 92.5% | — | ~26,000 ms | ~$3.00 |
| **ImageJev (CLIP)** | **92.5%** | **p = 0.23** *(indistinguishable)* | **~120 ms** | **~$0.004** |

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
