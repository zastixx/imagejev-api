"""Strongly typed decision results for ImageJev.

Results inherit from dict / Mapping to maintain 100% backward compatibility
with raw JSON responses while enabling IDE autocomplete and dot-attribute access.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional


class DictLikeResult(dict):
    """A dictionary subclass that allows dot-notation attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class ChoiceResult(DictLikeResult):
    """Result for schema={"type": "choice"}.

    Fields:
        choice: The selected option name (str), or None if allow_none=True and image is irrelevant.
        probability: Softmax probability of the selected choice (0.0 to 1.0).
        confidence: Normalized confidence margin (0.0 to 1.0).
        is_relevant: True if the visual input matches in-domain candidate schema options.
        relevance: Cosine similarity alignment score (0.0 to 1.0).
        probabilities: Mapping of all option names to their respective probabilities.
    """

    choice: Optional[str]
    probability: float
    confidence: float
    is_relevant: bool
    relevance: float
    probabilities: Dict[str, float]


class ScoreResult(DictLikeResult):
    """Result for schema={"type": "score"}.

    Fields:
        score: Continuous weighted rating (0.0 to N-1).
        probability: Probability of the peak discrete level.
        confidence: Confidence sharpness (0.0 to 1.0).
        dispersion: Mass outside the peak level and its immediate neighbors (0.0 to 1.0).
        legend: Mapping of level index to level description string.
        probabilities: Mapping of level index to its probability.
    """

    score: float
    probability: float
    confidence: float
    dispersion: float
    legend: Dict[int, str]
    probabilities: Dict[int, float]


class NoulResult(DictLikeResult):
    """Result for schema={"type": "noul"}.

    Matches standard binary Noul specification.

    Fields:
        probability: Probability of truth (0.0 to 1.0).
    """

    probability: float
