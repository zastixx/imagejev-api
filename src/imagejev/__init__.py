"""ImageJev: The image-native structured decision engine.

    Jev:       TEXT   + DECISION SCHEMA -> structured decision
    ImageJev:  IMAGE  + DECISION + SCHEMA -> structured decision
"""
from __future__ import annotations

__version__ = "0.1.2"

from .backends import Backend, MockBackend, OpenCLIPBackend
from .core import ImageJev
from .types import ChoiceResult, NoulResult, ScoreResult
from .client import ImageJevClient
from .errors import (
    ImageJevAuthenticationError,
    ImageJevError,
    ImageJevHTTPError,
    ImageJevInputError,
    ImageJevNotFoundError,
    ImageJevPayloadTooLargeError,
    ImageJevRateLimitError,
    ImageJevResponseError,
    ImageJevServerError,
    ImageJevTimeoutError,
    ImageJevTransportError,
    ImageJevUnavailableError,
    ImageJevValidationError,
)

__all__ = [
    "ImageJev",
    "Backend",
    "OpenCLIPBackend",
    "MockBackend",
    "ChoiceResult",
    "ScoreResult",
    "NoulResult",
    "ImageJevClient",
    "ImageJevError",
    "ImageJevHTTPError",
    "ImageJevInputError",
    "ImageJevAuthenticationError",
    "ImageJevNotFoundError",
    "ImageJevPayloadTooLargeError",
    "ImageJevRateLimitError",
    "ImageJevResponseError",
    "ImageJevServerError",
    "ImageJevTimeoutError",
    "ImageJevTransportError",
    "ImageJevUnavailableError",
    "ImageJevValidationError",
    "__version__",
]
