"""Interactive Local Image Tester for ImageJev.

Easily test any local image from your computer with structured closed-schema decisions!

Usage:
    py -3.11 examples/interactive_local_image.py
    or:
    py -3.11 examples/interactive_local_image.py "C:\\path\\to\\my_image.jpg"
"""
import sys
import os
import json
import time
from pathlib import Path
from imagejev import ImageJev

PRESETS = {
    "1": {
        "title": "E-Commerce Product Listing Moderation",
        "decision": "Is this product image suitable for listing on an online store?",
        "schema": {
            "type": "choice",
            "options": {
                "approve": "a clean, well-lit, in-focus studio product photo on neutral background",
                "reject": "a blurry, dark, damaged, torn box, or low quality unusable photo"
            }
        }
    },
    "2": {
        "title": "Scene & Subject Classification",
        "decision": "What is the primary subject in this photo?",
        "schema": {
            "type": "choice",
            "options": {
                "person": "a portrait or photo of a person or human",
                "pet": "a domestic dog, cat, or animal",
                "vehicle": "a car, motorcycle, truck, or vehicle",
                "food": "a dish, meal, drink, or food item",
                "nature": "an outdoor landscape of trees, mountains, or water",
                "shoes": "wearable item worn on feet"
            }
        }
    },
    "3": {
        "title": "Photo Quality & Lighting Score (0 to 3)",
        "decision": "Rate the overall lighting and clarity quality",
        "schema": {
            "type": "score",
            "levels": [
                "severely underexposed, dark, or out-of-focus blur",
                "dim or grainy amateur photo",
                "clear, well-exposed, balanced photograph",
                "crystal clear, bright, professional studio lighting"
            ]
        }
    },
    "4": {
        "title": "Binary Fact Verification (Noul)",
        "decision": "Is there a dog, cat, or pet visible in this image?",
        "schema": {
            "type": "noul",
            "true": "a domestic pet, dog, or cat is clearly visible",
            "false": "no pets or animals are present"
        }
    }
}

def main():
    print("=" * 80)
    print("IMAGEJEV: INTERACTIVE LOCAL IMAGE TESTER")
    print("=" * 80)

    # 1. Get image path
    if len(sys.argv) > 1:
        img_input = sys.argv[1].strip("\"'")
    else:
        # Check if test_images exist as defaults
        workspace_test_images = Path(__file__).resolve().parent.parent.parent / "test_images"
        default_samples = list(workspace_test_images.glob("*.jpg")) if workspace_test_images.exists() else []
        
        print("\nEnter the path to your local image file (or drag & drop a file here).")
        if default_samples:
            print(f"Sample available: {default_samples[0]}")
        
        img_input = input("\nImage Path > ").strip("\"'")
        if not img_input and default_samples:
            img_input = str(default_samples[0])

    img_path = Path(img_input)
    if not img_path.is_file():
        print(f"\n[Error] File not found: '{img_input}'")
        print("Please provide a valid path to an image (.jpg, .png, .webp, etc.).")
        return

    print(f"\nLoaded image: {img_path.name} ({img_path.stat().st_size / 1024:.1f} KB)")

    # 2. Select Decision Preset or Custom
    print("\nSelect a decision preset:")
    for k, v in PRESETS.items():
        print(f"  [{k}] {v['title']}")
    print("  [5] Custom Question & Schema")

    choice = input("\nChoose option [1-5, default 1] > ").strip() or "1"

    if choice in PRESETS:
        preset = PRESETS[choice]
        decision = preset["decision"]
        schema = preset["schema"]
    else:
        decision = input("\nEnter decision question > ").strip() or "What is in this picture?"
        print("Enter options separated by comma (e.g. cat, dog, car) or full JSON:")
        opts_input = input("Options > ").strip()
        if opts_input.startswith("{"):
            schema = json.loads(opts_input)
        else:
            opts = [o.strip() for o in opts_input.split(",") if o.strip()] or ["yes", "no"]
            schema = {"type": "choice", "options": {o: None for o in opts}}

    # 3. Execute Decision
    print("\n[Running Decision Engine with OpenCLIP ViT-B-32...]")
    t0 = time.perf_counter()
    jev = ImageJev()
    
    t_start = time.perf_counter()
    result = jev.decide(img_path, decision, schema)
    latency_ms = (time.perf_counter() - t_start) * 1000

    # 4. Display Results
    print("\n" + "=" * 80)
    print("STRUCTURED DECISION RESULT")
    print("=" * 80)
    print(f"Decision:     '{decision}'")
    print(f"Schema Type:  {schema['type'].upper()}")
    print(f"Inference:    {latency_ms:.1f} ms\n")

    if schema["type"] == "choice":
        print(f"SELECTED:     >>> {result.choice.upper()} <<<")
        print(f"Probability:  {result.probability:.4f} ({result.probability * 100:.1f}%)")
        print(f"Confidence:   {result.confidence:.4f}")
        print("\nAll Probabilities:")
        for opt, prob in sorted(result.probabilities.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * int(prob * 30)
            print(f"  {opt:<16} {prob:6.2%}  [{bar:<30}]")

    elif schema["type"] == "score":
        print(f"SCORE:        >>> {result.score:.2f} / {len(schema['levels']) - 1}.00 <<<")
        print(f"Peak Level:   {result.legend[max(result.probabilities, key=result.probabilities.get)]}")
        print(f"Confidence:   {result.confidence:.4f}")
        print(f"Dispersion:   {result.dispersion:.4f}")
        print("\nLevel Distribution:")
        for idx, prob in sorted(result.probabilities.items()):
            bar = "#" * int(prob * 30)
            print(f"  Level {idx} ({result.legend[idx]:<35}) {prob:6.2%}  [{bar:<30}]")

    elif schema["type"] == "noul":
        print(f"PROBABILITY:  >>> {result.probability:.4f} ({result.probability * 100:.1f}%) <<<")
        verdict = "TRUE (Yes)" if result.probability > 0.5 else "FALSE (No)"
        print(f"Verdict:      {verdict}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
