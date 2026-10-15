"""Synchronous client for a running ImageJev HTTP server."""
from __future__ import annotations

import base64
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from . import __version__
from .errors import (
    ImageJevAuthenticationError, ImageJevHTTPError, ImageJevInputError,
    ImageJevNotFoundError, ImageJevPayloadTooLargeError, ImageJevRateLimitError,
    ImageJevResponseError, ImageJevServerError, ImageJevTimeoutError,
    ImageJevTransportError, ImageJevUnavailableError, ImageJevValidationError,
)
from .types import ChoiceResult, NoulResult, ScoreResult


class ImageJevClient:
    """Client for the ImageJev REST API.

    The client owns a session it creates itself. A caller-supplied session is not
    closed by :meth:`close`.
    """

    def __init__(self, base_url: str, api_key: str | None = None, *,
                 timeout: float | tuple[float, float] = 30.0,
                 headers: Mapping[str, str] | None = None,
                 session: requests.Session | None = None,
                 multi_endpoint: str = "/v1/systemone") -> None:
        self.base_url = self._validate_base_url(base_url)
        if api_key is not None:
            if not isinstance(api_key, str) or "\r" in api_key or "\n" in api_key:
                raise ImageJevInputError("api_key must be a string without newlines")
        self.api_key = api_key
        self.timeout = self._validate_timeout(timeout)
        self._headers = self._validate_headers(headers)
        self._session = session or requests.Session()
        self._owns_session = session is None
        self.multi_endpoint = self._validate_endpoint(multi_endpoint, "multi_endpoint")

    @staticmethod
    def _validate_base_url(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ImageJevInputError("base_url must be a non-empty URL")
        value = value.strip().rstrip("/")
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.netloc or parts.query or parts.fragment:
            raise ImageJevInputError("base_url must be an http(s) URL without query or fragment")
        return value

    @staticmethod
    def _validate_timeout(value: Any) -> float | tuple[float, float]:
        if isinstance(value, bool):
            raise ImageJevInputError("timeout must be a positive finite number or a two-item tuple")
        if isinstance(value, (int, float)):
            if not math.isfinite(value) or value <= 0:
                raise ImageJevInputError("timeout must be a positive finite number or a two-item tuple")
            return float(value)
        if not isinstance(value, tuple) or len(value) != 2:
            raise ImageJevInputError("timeout must be a positive finite number or a two-item tuple")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in value):
            raise ImageJevInputError("timeout must be a positive finite number or a two-item tuple")
        return (float(value[0]), float(value[1]))

    @staticmethod
    def _validate_endpoint(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.startswith("/") or "?" in value or "#" in value:
            raise ImageJevInputError(f"{name} must be an absolute path without query or fragment")
        return value

    @staticmethod
    def _validate_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
        if headers is None:
            return {}
        if not isinstance(headers, Mapping):
            raise ImageJevInputError("headers must be a mapping of strings")
        result: dict[str, str] = {}
        for key, value in headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ImageJevInputError("header names and values must be strings")
            if "\r" in key or "\n" in key or "\r" in value or "\n" in value:
                raise ImageJevInputError("headers cannot contain newlines")
            result[key] = value
        return result

    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _request_headers(self, headers: Mapping[str, str] | None) -> dict[str, str]:
        merged: dict[str, str] = {"Accept": "application/json", "Content-Type": "application/json",
                                  "User-Agent": f"imagejev-python/{__version__}"}
        for source in (self._headers, self._validate_headers(headers)):
            for key, value in source.items():
                existing = next((k for k in merged if k.lower() == key.lower()), None)
                merged[existing or key] = value
        if self.api_key is not None:
            key = next((k for k in merged if k.lower() == "x-api-key"), "X-API-Key")
            merged[key] = self.api_key
        return merged

    @staticmethod
    def _image_value(image: Any) -> str:
        if isinstance(image, (bytes, bytearray)):
            return base64.b64encode(bytes(image)).decode("ascii")
        if isinstance(image, os.PathLike):
            try:
                return base64.b64encode(Path(image).read_bytes()).decode("ascii")
            except OSError as exc:
                raise ImageJevInputError(f"cannot read image path: {image}") from exc
        if isinstance(image, str):
            return image
        raise ImageJevInputError("image must be a string, bytes, or path-like object")

    @staticmethod
    def _header_number(headers: Mapping[str, str], name: str, integer: bool = False) -> float | int | None:
        raw = next((value for key, value in headers.items() if key.lower() == name.lower()), None)
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return int(value) if integer else value

    @staticmethod
    def _error_fields(body: Any, headers: Mapping[str, str]) -> tuple[Any, str | None, str | None]:
        detail = body.get("detail") if isinstance(body, dict) and "detail" in body else body
        code: str | None = None
        message: str | None = None
        if isinstance(body, dict):
            code_value = body.get("code")
            if isinstance(code_value, str):
                code = code_value
        if isinstance(detail, dict):
            detail_code = detail.get("error") or detail.get("code")
            detail_message = detail.get("message")
            if isinstance(detail_code, str):
                code = code or detail_code
            if isinstance(detail_message, str):
                message = detail_message
        elif isinstance(detail, str):
            message = detail
        request_id = next(
            (value for key, value in headers.items() if key.lower() in {"x-request-id", "x-correlation-id"}),
            None,
        )
        return detail, code, request_id

    def _request(self, endpoint: str, payload: dict[str, Any], *, timeout: Any = None,
                 headers: Mapping[str, str] | None = None) -> Any:
        url = self._url(self._validate_endpoint(endpoint, "endpoint"))
        effective_timeout = self.timeout if timeout is None else self._validate_timeout(timeout)
        try:
            response = self._session.request(
                "POST",
                url,
                json=payload,
                headers=self._request_headers(headers),
                timeout=effective_timeout,
            )
        except requests.Timeout as exc:
            raise ImageJevTimeoutError(f"request timed out: {url}") from exc
        except requests.RequestException as exc:
            raise ImageJevTransportError(f"request failed: {url}") from exc
        except (TypeError, ValueError) as exc:
            raise ImageJevInputError("request payload must be JSON serializable") from exc

        try:
            try:
                body = response.json()
            except ValueError:
                body = response.text
            status_code = response.status_code
            response_headers = dict(response.headers)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

        if not 200 <= status_code < 300:
            detail, code, request_id = self._error_fields(body, response_headers)
            kwargs = {
                "status_code": status_code,
                "method": "POST",
                "url": url,
                "detail": detail,
                "headers": response_headers,
                "code": code,
                "request_id": request_id,
            }
            if status_code in (401, 403):
                raise ImageJevAuthenticationError(**kwargs)
            if status_code == 413:
                raise ImageJevPayloadTooLargeError(**kwargs)
            if status_code in (400, 422):
                raise ImageJevValidationError(**kwargs)
            if status_code == 404:
                raise ImageJevNotFoundError(**kwargs)
            if status_code == 429:
                retry_after = self._header_number(response_headers, "Retry-After")
                if retry_after is None and isinstance(detail, dict):
                    raw_retry = detail.get("retry_after")
                    try:
                        retry_after = float(raw_retry) if raw_retry is not None else None
                    except (TypeError, ValueError):
                        retry_after = None
                limit_value = self._header_number(response_headers, "X-RateLimit-Limit", integer=True)
                remaining_value = self._header_number(response_headers, "X-RateLimit-Remaining", integer=True)
                reset_value = self._header_number(response_headers, "X-RateLimit-Reset", integer=True)
                limit = int(limit_value) if limit_value is not None else None
                remaining = int(remaining_value) if remaining_value is not None else None
                reset = int(reset_value) if reset_value is not None else None
                raise ImageJevRateLimitError(
                    **kwargs,
                    retry_after=retry_after,
                    limit=limit,
                    remaining=remaining,
                    reset=reset,
                )
            if status_code == 503:
                raise ImageJevUnavailableError(**kwargs)
            if status_code >= 500:
                raise ImageJevServerError(**kwargs)
            raise ImageJevHTTPError(**kwargs)
        if not isinstance(body, (dict, list)):
            raise ImageJevResponseError(f"expected JSON object or list from {url}")
        return body

    @staticmethod
    def _result(data: Any, schema_type: str) -> ChoiceResult | ScoreResult | NoulResult:
        if not isinstance(data, Mapping):
            raise ImageJevResponseError("result must be a JSON object")
        data = dict(data)
        classes = {"choice": ChoiceResult, "score": ScoreResult, "noul": NoulResult}
        required = {"choice": {"choice", "probability", "confidence", "probabilities"},
                    "score": {"score", "probability", "confidence", "dispersion", "legend", "probabilities"},
                    "noul": {"probability"}}
        if schema_type not in classes or not required[schema_type].issubset(data):
            raise ImageJevResponseError(f"response does not match {schema_type!r} result shape")
        if schema_type == "choice":
            if "is_relevant" not in data or "relevance" not in data:
                raise ImageJevResponseError("choice response is missing relevance fields")
            if not isinstance(data["probabilities"], Mapping):
                raise ImageJevResponseError("choice probabilities must be a JSON object")
        if schema_type == "score":
            if not isinstance(data["legend"], Mapping) or not isinstance(data["probabilities"], Mapping):
                raise ImageJevResponseError("score legend and probabilities must be JSON objects")
            try:
                data["legend"] = {int(k): v for k, v in data["legend"].items()}
                data["probabilities"] = {int(k): v for k, v in data["probabilities"].items()}
            except (TypeError, ValueError) as exc:
                raise ImageJevResponseError("score probability keys must be integers") from exc
        return classes[schema_type](**data)

    @staticmethod
    def _schema_type(schema: Mapping[str, Any]) -> str:
        if not isinstance(schema, Mapping):
            raise ImageJevInputError("schema must be a mapping")
        schema_type = schema.get("type")
        if schema_type not in {"choice", "score", "noul"}:
            raise ImageJevInputError("schema['type'] must be 'choice', 'score', or 'noul'")
        return schema_type

    def decide(self, image: Any, decision: str, schema: Mapping[str, Any], *, timeout: Any = None,
               headers: Mapping[str, str] | None = None) -> ChoiceResult | ScoreResult | NoulResult:
        if not isinstance(decision, str):
            raise ImageJevInputError("decision must be a string")
        if not isinstance(schema, Mapping):
            raise ImageJevInputError("schema must be a mapping")
        schema = dict(schema)
        schema_type = self._schema_type(schema)
        data = self._request("/decide", {"image": self._image_value(image), "decision": decision, "schema": schema}, timeout=timeout, headers=headers)
        return self._result(data, schema_type)

    def decide_batch(self, images: Sequence[Any], decision: str, schema: Mapping[str, Any], *, timeout: Any = None,
                     headers: Mapping[str, str] | None = None) -> list[ChoiceResult | ScoreResult | NoulResult]:
        if not isinstance(decision, str):
            raise ImageJevInputError("decision must be a string")
        if not isinstance(schema, Mapping):
            raise ImageJevInputError("schema must be a mapping")
        schema = dict(schema)
        schema_type = self._schema_type(schema)
        if isinstance(images, (str, bytes, bytearray)):
            raise ImageJevInputError("images must be a sequence of image references")
        try:
            image_list = list(images)
        except TypeError as exc:
            raise ImageJevInputError("images must be a sequence of image references") from exc
        if not image_list:
            raise ImageJevInputError("images cannot be empty")
        data = self._request("/decide_batch", {"images": [self._image_value(i) for i in image_list], "decision": decision, "schema": dict(schema)}, timeout=timeout, headers=headers)
        if not isinstance(data, list):
            raise ImageJevResponseError("batch response must be a JSON list")
        if len(data) != len(image_list):
            raise ImageJevResponseError("batch response length does not match request")
        return [self._result(item, schema_type) for item in data]

    def decide_multi(self, image: Any, questions: Mapping[str, Mapping[str, Any]], *, timeout: Any = None,
                     headers: Mapping[str, str] | None = None) -> dict[str, ChoiceResult | ScoreResult | NoulResult]:
        if not isinstance(questions, Mapping):
            raise ImageJevInputError("questions must be a mapping")
        question_map = dict(questions)
        if not question_map:
            raise ImageJevInputError("questions cannot be empty")
        for key, spec in question_map.items():
            if not isinstance(key, str) or not isinstance(spec, Mapping):
                raise ImageJevInputError("questions must map string IDs to schema mappings")
        data = self._request(self.multi_endpoint, {"image": self._image_value(image), "questions": question_map}, timeout=timeout, headers=headers)
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise ImageJevResponseError("multi response must contain an answers object")
        if set(data["answers"]) != set(question_map):
            raise ImageJevResponseError("multi response question IDs do not match the request")
        answers = {}
        for key, value in data["answers"].items():
            spec = question_map[key]
            schema = spec.get("schema") or spec
            if not isinstance(schema, Mapping):
                raise ImageJevInputError(f"question {key!r} schema must be a mapping")
            answers[key] = self._result(value, self._schema_type(schema))
        return answers

    def systemone(self, image: Any, questions: Mapping[str, Mapping[str, Any]], *, timeout: Any = None,
                  headers: Mapping[str, str] | None = None) -> dict[str, ChoiceResult | ScoreResult | NoulResult]:
        """Compatibility alias for :meth:`decide_multi`."""
        return self.decide_multi(image, questions, timeout=timeout, headers=headers)

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "ImageJevClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
