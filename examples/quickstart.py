"""Quickstart Example: Zero-Shot Image Decision using ImageJev SDK."""
from imagejev import ImageJev

def main():
    print("Initializing ImageJev...")
    jev = ImageJev()

    decision = "What type of scene or object is shown?"
    schema = {
        "type": "choice",
        "options": {
            "nature": "a natural landscape with mountains, trees, or water",
            "urban": "a city street with buildings, roads, and cars",
            "indoor": "an interior room with furniture",
            "food": "a delicious plate or dish of food"
        }
    }

    # Pass local path, URL, PIL image, or bytes
    image_url = "https://images.unsplash.com/photo-1506744038136-46273834b3fb"
    print(f"Making decision on image: {image_url}...")
    
    result = jev.decide(image_url, decision, schema)

    print("\n--- Decision Result ---")
    print(f"Choice:        {result.choice}")
    print(f"Probability:   {result.probability:.4f} ({result.probability * 100:.1f}%)")
    print(f"Confidence:    {result.confidence:.4f}")
    print(f"Probabilities: {result.probabilities}")

if __name__ == "__main__":
    main()
