"""Offline unit tests for ImageJev using MockBackend."""
import warnings
import pytest
from imagejev import ImageJev, MockBackend, ChoiceResult, ScoreResult, NoulResult


@pytest.fixture
def mock_engine():
    return ImageJev(backend=MockBackend(seed=42))


def test_choice_basic_and_types(mock_engine):
    schema = {
        "type": "choice",
        "options": {
            "cat": "a cute cat sitting on a mat",
            "dog": "a friendly golden retriever dog",
            "car": "a fast sports car on a highway"
        }
    }
    res = mock_engine.decide("cute cat sitting on a mat", "What is in the image?", schema)

    assert isinstance(res, ChoiceResult)
    assert res.choice == "cat"
    assert 0.0 <= res.probability <= 1.0
    assert 0.0 <= res.confidence <= 1.0
    assert set(res.probabilities.keys()) == {"cat", "dog", "car"}
    assert abs(sum(res.probabilities.values()) - 1.0) < 1e-5
    # Dict backward compatibility
    assert res["choice"] == "cat"
    assert "probability" in res


def test_choice_bare_judgment_words_warn(mock_engine):
    schema = {"type": "choice", "options": ["yes", "no"]}
    with pytest.warns(UserWarning, match="judgment words"):
        res = mock_engine.decide("clean photo", "Is this suitable?", schema)
    assert res.choice in ("yes", "no")


def test_score_properties_and_bounds(mock_engine):
    schema = {
        "type": "score",
        "levels": ["poor", "fair", "good", "excellent"]
    }
    res = mock_engine.decide("a good quality clear image", "Quality score", schema)

    assert isinstance(res, ScoreResult)
    assert 0.0 <= res.score <= 3.0
    assert 0.0 <= res.probability <= 1.0
    assert 0.0 <= res.confidence <= 1.0
    assert 0.0 <= res.dispersion <= 1.0
    assert res.legend == {0: "poor", 1: "fair", 2: "good", 3: "excellent"}
    assert len(res.probabilities) == 4

    # Bounds validation (<2 or >10 levels)
    with pytest.raises(ValueError, match="requires 2-10 levels"):
        mock_engine.decide("img", "rate", {"type": "score", "levels": ["one"]})

    with pytest.raises(ValueError, match="requires 2-10 levels"):
        mock_engine.decide("img", "rate", {"type": "score", "levels": [f"lvl_{i}" for i in range(11)]})


def test_noul_schema_and_criteria(mock_engine):
    schema = {
        "type": "noul",
        "true": "a photo of a cat",
        "false": "a photo with no cats or pets"
    }
    res = mock_engine.decide("a photo of a cat", "Check cat", schema)

    assert isinstance(res, NoulResult)
    assert 0.0 <= res.probability <= 1.0
    # Jev spec compliance: Noul has no confidence field
    assert "confidence" not in res
    assert not hasattr(res, "confidence")


def test_invalid_schema_type_raises(mock_engine):
    with pytest.raises(ValueError, match='schema\\["type"\\] must be "choice", "score", or "noul"'):
        mock_engine.decide("img", "decision", {"type": "unsupported_type"})
