"""Tests for Open-Set / Out-Of-Distribution (OOD) rejection and relevance gating."""
from pathlib import Path
from PIL import Image
import pytest

from imagejev import ImageJev
from imagejev.backends import MockBackend, OpenCLIPBackend

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TEST_IMAGES_DIR = ROOT_DIR / "test_images"


def test_mock_backend_open_set():
    """Test open-set schema behavior on MockBackend."""
    engine = ImageJev(backend=MockBackend())

    # In-domain match
    schema = {
        "type": "choice",
        "options": {"cat": "a cat", "dog": "a dog"},
        "allow_none": True,
        "relevance_threshold": 0.01
    }
    res = engine.decide("cat photo", "What animal?", schema)
    assert res.is_relevant is True
    assert res.relevance > 0.0
    assert res.choice in ("cat", "dog")


def test_real_open_set_ood_rejection():
    """Test real OpenCLIP engine rejecting out-of-domain images with allow_none=True."""
    dog_path = TEST_IMAGES_DIR / "golden_retriever.jpg"
    car_path = TEST_IMAGES_DIR / "sports_car.jpg"
    pizza_path = TEST_IMAGES_DIR / "pizza.jpg"

    if not dog_path.is_file() or not car_path.is_file():
        pytest.skip("Test images not found")

    engine = ImageJev(model="ViT-B-32", device="cpu")

    # Schema only defines pets: cat vs dog
    pet_schema_open = {
        "type": "choice",
        "options": {
            "cat": "a photo of a cat feline",
            "dog": "a photo of a dog canine golden retriever"
        },
        "allow_none": True,
        "relevance_threshold": 0.20
    }

    # 1. In-domain image (dog) -> MUST select dog
    dog_res = engine.decide(dog_path, "Which pet?", pet_schema_open)
    assert dog_res.is_relevant is True
    assert dog_res.choice == "dog"
    assert dog_res.relevance >= 0.25

    # 2. Out-of-domain image (sports car) -> MUST reject (choice=None)
    car_res = engine.decide(car_path, "Which pet?", pet_schema_open)
    assert car_res.is_relevant is False
    assert car_res.choice is None
    assert car_res.relevance < 0.20

    # 3. Out-of-domain image (pizza) -> MUST reject (choice=None)
    if pizza_path.is_file():
        pizza_res = engine.decide(pizza_path, "Which pet?", pet_schema_open)
        assert pizza_res.is_relevant is False
        assert pizza_res.choice is None
        assert pizza_res.relevance < 0.20


def test_real_open_set_batch():
    """Test batch inference with mixed in-domain and out-of-domain images."""
    dog_path = TEST_IMAGES_DIR / "golden_retriever.jpg"
    car_path = TEST_IMAGES_DIR / "sports_car.jpg"
    mug_path = TEST_IMAGES_DIR / "coffee_mug.jpg"

    if not all(p.is_file() for p in [dog_path, car_path, mug_path]):
        pytest.skip("Test images not found")

    engine = ImageJev(model="ViT-B-32", device="cpu")

    pet_schema_open = {
        "type": "choice",
        "options": {
            "cat": "a photo of a cat",
            "dog": "a photo of a dog canine retriever"
        },
        "allow_none": True,
        "relevance_threshold": 0.20
    }

    results = engine.decide_batch([dog_path, car_path, mug_path], "Which pet?", pet_schema_open)
    assert len(results) == 3

    # Dog -> matched
    assert results[0].choice == "dog"
    assert results[0].is_relevant is True

    # Car -> rejected
    assert results[1].choice is None
    assert results[1].is_relevant is False

    # Mug -> rejected
    assert results[2].choice is None
    assert results[2].is_relevant is False
