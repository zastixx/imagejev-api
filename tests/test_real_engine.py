"""Integration tests with real OpenCLIP backend."""
from pathlib import Path
from PIL import Image
import pytest

from imagejev import ImageJev, OpenCLIPBackend, ChoiceResult, ScoreResult, NoulResult

WORKSPACE_TEST_IMAGES = Path(__file__).resolve().parent.parent.parent / "test_images"


@pytest.fixture(scope="module")
def real_engine():
    return ImageJev(OpenCLIPBackend())


def test_real_clip_choice_with_pil_and_path(real_engine):
    # Create test image or load from workspace
    img_path = WORKSPACE_TEST_IMAGES / "golden_retriever.jpg"
    if not img_path.exists():
        # Fallback synthetic PIL test image
        img = Image.new("RGB", (224, 224), color=(240, 200, 100))
    else:
        img = img_path

    schema = {
        "type": "choice",
        "options": {
            "dog": "a golden retriever dog portrait",
            "car": "a red sports car on road",
            "coffee": "a cup of coffee"
        }
    }
    res = real_engine.decide(img, "Identify the subject", schema)
    assert isinstance(res, ChoiceResult)
    if img_path.exists():
        assert res.choice == "dog"
        assert res.probability > 0.8
        assert res.confidence > 0.7


def test_real_clip_batch_decide(real_engine):
    img_path_dog = WORKSPACE_TEST_IMAGES / "golden_retriever.jpg"
    img_path_car = WORKSPACE_TEST_IMAGES / "sports_car.jpg"

    if img_path_dog.exists() and img_path_car.exists():
        images = [img_path_dog, img_path_car]
        schema = {
            "type": "choice",
            "options": {
                "dog": "a golden retriever dog portrait",
                "car": "a red sports car on road",
            }
        }
        results = real_engine.decide_batch(images, "Identify subject", schema)
        assert len(results) == 2
        assert results[0].choice == "dog"
        assert results[1].choice == "car"
