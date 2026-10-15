"""Exceptions raised by the ImageJev HTTP client."""
from __future__ import annotations

from typing import Any, Mapping


class ImageJevError(Exception):
    """Base class for client errors."""


class ImageJevInputError(ImageJevError, ValueError):
    """The client configuration or request payload is invalid."""


class ImageJevTransportError(ImageJevError):
    """The request could not reach the server."""


class ImageJevTimeoutError(ImageJevTransportError, TimeoutError):
    """The request exceeded its connect or read timeout."""


class ImageJevResponseError(ImageJevError):
    """The server returned a successful response with an invalid shape."""


class ImageJevHTTPError(ImageJevError):
    """The server returned a non-success HTTP response."""

    def __init__(self, status_code: int, method: str, url: str, detail: Any = None,
                 headers: Mapping[str, str] | None = None, code: str | None = None,
                 request_id: str | None = None):
        self.status_code = status_code
        self.method = method
        self.url = url
        self.detail = detail
        self.code = code
        self.request_id = request_id
        self.response_headers = dict(headers or {})
        if isinstance(detail, str) and detail:
            message = detail
        else:
            message = f"HTTP {status_code} from {method} {url}"
        super().__init__(str(message))


class ImageJevAuthenticationError(ImageJevHTTPError):
    pass


class ImageJevValidationError(ImageJevHTTPError):
    pass


class ImageJevPayloadTooLargeError(ImageJevValidationError):
    pass


class ImageJevRateLimitError(ImageJevHTTPError):
    def __init__(self, *args: Any, retry_after: float | None = None,
                 limit: int | None = None, remaining: int | None = None,
                 reset: int | None = None, **kwargs: Any):
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining
        self.reset = reset
        super().__init__(*args, **kwargs)


class ImageJevNotFoundError(ImageJevHTTPError):
    pass


class ImageJevServerError(ImageJevHTTPError):
    pass


class ImageJevUnavailableError(ImageJevServerError):
    pass
