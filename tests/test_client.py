import base64

import pytest
import requests

from imagejev import (
    ChoiceResult,
    ImageJevClient,
    NoulResult,
    ScoreResult,
)
from imagejev.errors import (
    ImageJevAuthenticationError,
    ImageJevRateLimitError,
    ImageJevResponseError,
    ImageJevTimeoutError,
    ImageJevValidationError,
)


class Response:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = str(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    def close(self):
        pass


class Session:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.closed = False

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response

    def close(self):
        self.closed = True


def test_decide_sends_wire_schema_and_returns_typed_result():
    session = Session(Response(body={"choice": "cat", "probability": .9, "confidence": .8,
                                    "is_relevant": True, "relevance": .7, "probabilities": {"cat": .9}}))
    client = ImageJevClient("https://api.example.test/prefix/", api_key="secret", session=session)
    result = client.decide("cat.jpg", "What is shown?", {"type": "choice", "options": {"cat": "a cat"}})

    assert isinstance(result, ChoiceResult)
    assert result.choice == "cat"
    args, kwargs = session.calls[0]
    assert args[0] == "POST"
    assert args[1] == "https://api.example.test/prefix/decide"
    assert kwargs["json"]["schema"]["type"] == "choice"
    assert "schema_" not in kwargs["json"]
    assert kwargs["headers"]["X-API-Key"] == "secret"


def test_bytes_are_base64_encoded_and_score_keys_are_normalized():
    session = Session(Response(body={"score": 1.2, "probability": .7, "confidence": .6,
                                    "dispersion": .1, "legend": {"0": "low", "1": "high"},
                                    "probabilities": {"0": .3, "1": .7}}))
    client = ImageJevClient("http://localhost", session=session)
    result = client.decide(b"image", "Rate it", {"type": "score", "levels": ["low", "high"]})
    assert isinstance(result, ScoreResult)
    assert result.legend == {0: "low", 1: "high"}
    assert session.calls[0][1]["json"]["image"] == base64.b64encode(b"image").decode()


def test_batch_preserves_order_and_uses_batch_endpoint():
    result = {"choice": "cat", "probability": .9, "confidence": .8,
              "is_relevant": True, "relevance": .7, "probabilities": {"cat": .9}}
    session = Session(Response(body=[result, result]))
    client = ImageJevClient("https://api.example.test", session=session)
    results = client.decide_batch(["one", "two"], "Classify", {"type": "choice"})

    assert [item.choice for item in results] == ["cat", "cat"]
    assert session.calls[0][0][0] == "POST"
    assert session.calls[0][0][1] == "https://api.example.test/decide_batch"
    assert session.calls[0][1]["json"]["images"] == ["one", "two"]


def test_systemone_alias_and_nested_schema_are_supported():
    session = Session(Response(body={"answers": {"q": {"probability": .4}}}))
    client = ImageJevClient("https://api.example.test", session=session)
    result = client.systemone("img", {"q": {"schema": {"type": "noul"}}})

    assert isinstance(result["q"], NoulResult)
    assert session.calls[0][0][1] == "https://api.example.test/v1/systemone"


def test_multi_unwraps_answers_and_context_closes_owned_session():
    body = {"answers": {"q": {"probability": .4}}}
    session = Session(Response(body=body))
    with ImageJevClient("http://localhost", session=session) as client:
        result = client.decide_multi("img", {"q": {"type": "noul"}})
    assert isinstance(result["q"], NoulResult)
    assert not session.closed  # injected sessions remain caller-owned


@pytest.mark.parametrize("status, error", [(401, ImageJevAuthenticationError), (422, ImageJevValidationError)])
def test_http_errors_are_typed(status, error):
    session = Session(Response(status=status, body={"detail": "bad request"}))
    client = ImageJevClient("http://localhost", session=session)
    with pytest.raises(error) as raised:
        client.decide("img", "q", {"type": "noul"})
    assert raised.value.status_code == status


def test_rate_limit_error_keeps_retry_and_limit_headers():
    session = Session(Response(
        status=429,
        body={"detail": {"error": "rate_limit_exceeded", "retry_after": 7}},
        headers={
            "Retry-After": "7",
            "X-RateLimit-Limit": "3",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "123",
        },
    ))
    client = ImageJevClient("http://localhost", session=session)

    with pytest.raises(ImageJevRateLimitError) as raised:
        client.decide("img", "q", {"type": "noul"})

    assert raised.value.retry_after == 7
    assert raised.value.limit == 3
    assert raised.value.remaining == 0
    assert raised.value.reset == 123


def test_timeout_is_wrapped_and_caller_headers_are_not_mutated():
    class TimeoutSession(Session):
        def request(self, *args, **kwargs):
            raise requests.Timeout("read timed out")

    headers = {"X-Trace-ID": "trace-1"}
    client = ImageJevClient("http://localhost", headers=headers, session=TimeoutSession(Response()))
    with pytest.raises(ImageJevTimeoutError):
        client.decide("img", "q", {"type": "noul"}, timeout=(1, 2), headers={"X-Request-ID": "request-1"})
    assert headers == {"X-Trace-ID": "trace-1"}


def test_malformed_success_is_rejected():
    client = ImageJevClient("http://localhost", session=Session(Response(body={"choice": "cat"})))
    with pytest.raises(ImageJevResponseError):
        client.decide("img", "q", {"type": "choice"})
