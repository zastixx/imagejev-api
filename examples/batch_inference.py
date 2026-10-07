"""High-Throughput Batch Inference Example using ImageJev SDK."""
import time
from pathlib import Path
from imagejev import ImageJev

def main():
    jev = ImageJev()

    # Batch of images
    image_paths = [
        "https://images.unsplash.com/photo-1543466835-00a7907e9de1",  # dog
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba",  # cat
        "https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7",  # car
    ]

    schema = {
        "type": "choice",
        "options": {
            "dog": "a photo of a dog",
            "cat": "a photo of a cat",
            "car": "a photo of a car",
        }
    }

    print(f"Executing batch decision on {len(image_paths)} images...")
    t0 = time.perf_counter()
    results = jev.decide_batch(image_paths, "Classify subject", schema)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"Batch completed in {elapsed_ms:.1f}ms ({elapsed_ms/len(image_paths):.1f}ms per image):\n")
    for img, res in zip(image_paths, results):
        print(f"Image: {img[:50]}...")
        print(f"  Choice:      {res.choice}")
        print(f"  Confidence:  {res.confidence:.3f}\n")

if __name__ == "__main__":
    main()
