"""Unit tests for high-throughput batch inference in ImageJev."""
import pytest
from imagejev import ImageJev, MockBackend, ChoiceResult, ScoreResult, NoulResult


@pytest.fixture
def mock_engine():
    return ImageJev(backend=MockBackend(seed=123))


def test_decide_batch_choice_equivalence(mock_engine):
    images = [
        "a cute cat sitting on a mat",
        "a fast sports car on a road",
        "a cup of hot morning coffee"
    ]
    schema = {
        "type": "choice",
        "options": {
            "cat": "a cute cat sitting on a mat",
            "car": "a fast sports car on a road",
            "coffee": "a cup of hot morning coffee"
        }
    }

    # Batch decide
    batch_res = mock_engine.decide_batch(images, "Classify object", schema)
    assert len(batch_res) == 3
    assert all(isinstance(r, ChoiceResult) for r in batch_res)

    # Compare with single decide
    for img, b_res in zip(images, batch_res):
        s_res = mock_engine.decide(img, "Classify object", schema)
        assert b_res.choice == s_res.choice
        assert abs(b_res.probability - s_res.probability) < 1e-5
        assert abs(b_res.confidence - s_res.confidence) < 1e-5


def test_decide_batch_score(mock_engine):
    images = ["poor dark photo", "good crisp photo"]
    schema = {"type": "score", "levels": ["poor dark photo", "good crisp photo"]}

    batch_res = mock_engine.decide_batch(images, "Rate photo", schema)
    assert len(batch_res) == 2
    assert all(isinstance(r, ScoreResult) for r in batch_res)


def test_decide_batch_noul(mock_engine):
    images = ["cat photo", "dog photo"]
    schema = {"type": "noul", "true": "cat photo", "false": "not a cat"}

    batch_res = mock_engine.decide_batch(images, "Is it a cat?", schema)
    assert len(batch_res) == 2
    assert all(isinstance(r, NoulResult) for r in batch_res)


def test_decide_batch_empty_list(mock_engine):
    res = mock_engine.decide_batch([], "any", {"type": "choice", "options": ["a", "b"]})
    assert res == []
