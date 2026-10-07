"""E-Commerce Product Image Listing Moderation with Confidence Gating."""
from imagejev import ImageJev

def moderate_product_image(image_input):
    jev = ImageJev()

    schema = {
        "type": "choice",
        "options": {
            "approve": "a clear, well-lit, high quality studio photo of a single product on clean background",
            "reject": "a blurry, damaged, dark, low-quality, or confusing photo unsuitable for a store"
        }
    }

    result = jev.decide(
        image_input,
        "Is this image suitable for listing on an e-commerce catalog?",
        schema
    )

    # Production Confidence Gating Pattern:
    # High confidence -> Auto-Action
    # Low confidence -> Flag for human review
    CONFIDENCE_THRESHOLD = 0.80

    if result.confidence < CONFIDENCE_THRESHOLD:
        action = "HUMAN_REVIEW_REQUIRED"
    elif result.choice == "approve":
        action = "AUTO_APPROVED"
    else:
        action = "AUTO_REJECTED"

    return {
        "action": action,
        "choice": result.choice,
        "confidence": result.confidence,
        "probability": result.probability
    }

if __name__ == "__main__":
    test_img = "https://images.unsplash.com/photo-1523275335684-37898b6baf30"
    print(f"Evaluating: {test_img}")
    decision = moderate_product_image(test_img)
    print(f"Result: {decision}")
