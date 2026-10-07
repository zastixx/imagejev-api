"""ImageJev: The image-native structured decision engine.

    Jev:       TEXT   + DECISION SCHEMA -> structured decision
    ImageJev:  IMAGE  + DECISION + SCHEMA -> structured decision
"""
from __future__ import annotations

from .backends import Backend, MockBackend, OpenCLIPBackend
from .core import ImageJev
from .types import ChoiceResult, NoulResult, ScoreResult

__version__ = "0.1.0"
__all__ = [
    "ImageJev",
    "Backend",
    "OpenCLIPBackend",
    "MockBackend",
    "ChoiceResult",
    "ScoreResult",
    "NoulResult",
    "__version__",
]
