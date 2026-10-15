# ImageJev

An image-native structured decision engine using frozen OpenCLIP embeddings.
Provide an image and a decision schema to obtain structured results without
autoregressive text generation. Structured output does not guarantee a correct
prediction, and ImageJev is not a general-purpose vision-language model.

The default backbone is OpenCLIP `ViT-B-32` with `laion2b_s34b_b79k` weights.

## Installation

```bash
pip install imagejev
pip install "imagejev[server]"  # Optional FastAPI server
```

The package uses PyTorch, OpenCLIP, Pillow, and NumPy. Real inference requires
model weights; the initial model load may download them. GPU availability depends
on your PyTorch installation and hardware.

## Quickstart

```python
from imagejev import ImageJev

jev = ImageJev()
result = jev.decide(
    image="product.jpg",
    decision="Is this product image suitable for listing?",
    schema={
        "type": "choice",
        "options": {
            "yes": "a clear, well-lit, in-focus product photo",
            "no": "a blurry, underexposed, cluttered, or damaged product photo",
        },
    },
)

print(result.choice)
print(result.probability)
print(result.confidence)
print(result.probabilities)
```

Outputs depend on the image, model, and option descriptions. No fixed prediction
or probability is implied by this example. For `choice` and `score`, the schema
descriptions drive the comparison; the decision question is not an additional
reasoning step.

## Decision schemas

- `choice`: compares described options and returns the selected option,
  probabilities, and normalized confidence. `allow_none=True` enables abstention
  using a null anchor and relevance threshold.
- `score`: accepts 2 to 10 ordered descriptions and returns a zero-based expected
  score, probabilities, confidence, and dispersion. Inspect the distribution and
  dispersion rather than relying only on the scalar score.
- `noul`: compares explicit `true` and `false` descriptions and returns a binary
  probability. It does not expose a separate confidence field.

```python
score = jev.decide(
    image="product.jpg",
    decision="Rate lighting and clarity",
    schema={"type": "score", "levels": [
        "an underexposed or blurry photograph",
        "a clearly exposed photograph",
        "a bright professional studio photograph",
    ]},
)

binary = jev.decide(
    image="product.jpg",
    decision="Is there a vehicle?",
    schema={
        "type": "noul",
        "true": "a photograph of a motor vehicle",
        "false": "a photograph without a motor vehicle",
    },
)
```

## Batch and multi-question inference

`decide_batch(images, decision, schema)` evaluates images against one schema.
`decide_multi(image, questions)` evaluates multiple schemas after encoding one
image. Batch throughput depends on hardware and batch size; larger batches are
not necessarily faster.

## HTTP server

```bash
imagejev-server --host 127.0.0.1 --port 8000
```

OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.
Endpoints include `/health`, `/decide`, `/decide_upload`, `/decide_batch`, and
`/v1/systemone` (also available as `/v1/decide`).

Server settings are read from environment variables. Authentication is disabled
unless `IMAGEJEV_API_KEYS` is configured; when configured, requests require the
`X-API-Key` header. Configurable defaults include 60 single-decision requests per
minute, 15 batch requests per minute, and 50 images per batch request. These are
request limits, not measured throughput guarantees. Review access controls and
deployment configuration before exposing the server publicly.

## Remote Python client

Use `ImageJevClient` when inference runs behind the HTTP server rather than in
the current Python process:

```python
from imagejev import ImageJevClient

with ImageJevClient("https://api.example.com", api_key="your-api-key") as client:
    result = client.decide(
        image="https://example.com/product.jpg",
        decision="Is this product image suitable for listing?",
        schema={
            "type": "choice",
            "options": {
                "yes": "a clear, well-lit product photograph",
                "no": "a blurry, cluttered, or damaged product photograph",
            },
        },
    )
    print(result.choice, result.confidence)
```

The client sends `X-API-Key`, uses a finite 30-second default timeout, and
returns the same `ChoiceResult`, `ScoreResult`, and `NoulResult` types as local
inference. It also provides `decide_batch()` and `decide_multi()`. HTTP failures,
transport failures, timeouts, and malformed responses raise typed exceptions
from `imagejev.errors`.

## Reported benchmark results

The following historical measurements are reported in the project's `data.md`
notes dated October 11, 2026. They have not been independently reproduced for this
description update. The raw CSV and comparison-result JSON artifacts referenced
by those notes were not located during this review. Treat these as reported
results, not independently verified release benchmarks or service guarantees.

### Live-API product-listing evaluation

| Measurement | Reported result |
|---|---|
| Agreement with proxy labels | 86.8% of 1,014 successful image requests |
| Agreement at confidence >= 0.95 | 95.5% across 580 images; 57% coverage |
| Requests excluded | 27 rate-limited requests out of 1,041 attempted |

Images came from six Unsplash search categories. Expected labels were inferred
from search queries, not verified by human annotators. These numbers measure
agreement with weak proxy labels, not audited real-world accuracy. They apply to
the tested product-listing schema, not arbitrary visual tasks. The dark/low-light
category had 74.1% agreement; deliberately moody lighting and underexposure can be
ambiguous under a combined quality schema.

### Controlled CIFAR-10 comparison

| System named in the benchmark notes | Reported accuracy on 200 images |
|---|---|
| ImageJev | 92.5% |
| Claude Haiku 4.5 | 80.0% |
| Claude Sonnet 5 | 95.0% |

The notes report McNemar p=0.23 for ImageJev versus Sonnet and a paired-bootstrap
95% confidence interval of -6.0 to +0.5 percentage points for the accuracy
difference (ImageJev minus Sonnet). Failure to detect a significant difference
does not establish equivalence or a statistical tie. This small CIFAR-10 sample
is not a head-to-head evaluation on product photographs. Exact model identities
and run details require the original result artifacts before independent audit.

### Compute time versus API latency

| Measurement | Reported result | Scope |
|---|---|---|
| T4 model-compute median | 6.51 ms | Dedicated-process compute benchmark |
| Live API round-trip median, run 1 | 405 ms | Includes network and source-image download |
| Live API round-trip median, run 2 | 613 ms | Includes network and source-image download |
| Peak tested T4 batch throughput | 309 images/second at batch size 32 | Compute benchmark, not API capacity |

Larger tested batches reduced throughput. These measurements are not a matched
end-to-end speed comparison against hosted VLMs. Hardware, image transfer,
concurrency, and cold starts affect user-visible latency.

### Cost

Measured production cost per decision is unresolved pending actual billed
execution-duration data. No fixed cost or cost-saving multiplier is claimed.

## Limitations and schema design

- Describe visual alternatives explicitly. Bare judgment labels such as `yes`
  and `no` are poor image-text comparison prompts.
- Confidence is derived from the model's option distribution, not a guarantee of
  correctness or an independently calibrated probability for your use case.
- Select routing thresholds using labeled validation data from your target task;
  send uncertain cases to human review or another model.
- Closed options can produce confident errors on unrelated images. Consider a
  catch-all option or configured abstention and validate its behavior.
- Embedded text, subtle defects, and ambiguous criteria require task-specific
  evaluation. ImageJev does not replace OCR or unrestricted visual reasoning.

## License

MIT License. See the LICENSE file included in the distribution.
