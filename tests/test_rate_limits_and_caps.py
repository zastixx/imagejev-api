"""Unit tests for Rate Limiting, Batch Size Caps, and Production Safeguards."""
import pytest
from fastapi.testclient import TestClient

from imagejev import ImageJev, MockBackend
from imagejev.server import (
    app,
    set_engine,
    rate_limiter,
    MAX_BATCH_SIZE,
    MAX_QUESTIONS,
    VALID_API_KEYS,
)


@pytest.fixture(autouse=True)
def setup_test_engine():
    set_engine(ImageJev(MockBackend()))
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def test_batch_size_cap_exceeded():
    """Verify that requests exceeding MAX_BATCH_SIZE are rejected with HTTP 400."""
    client = TestClient(app)
    too_many_images = [f"image_{i}.jpg" for i in range(MAX_BATCH_SIZE + 1)]

    payload = {
        "images": too_many_images,
        "decision": "Inspect",
        "schema": {"type": "choice", "options": {"ok": "good", "bad": "bad"}},
    }

    res = client.post("/decide_batch", json=payload)
    assert res.status_code == 400
    assert "exceeds the maximum allowed limit" in res.json()["detail"]


def test_batch_empty_rejected():
    """Verify that empty batch list is rejected with HTTP 400."""
    client = TestClient(app)
    payload = {
        "images": [],
        "decision": "Inspect",
        "schema": {"type": "choice", "options": {"ok": "good", "bad": "bad"}},
    }
    res = client.post("/decide_batch", json=payload)
    assert res.status_code == 400
    assert "cannot be empty" in res.json()["detail"]


def test_question_count_cap_exceeded():
    """Verify that multi-question requests exceeding MAX_QUESTIONS are rejected with HTTP 400."""
    client = TestClient(app)
    too_many_questions = {
        f"q_{i}": {"type": "choice", "options": {"a": "option a", "b": "option b"}}
        for i in range(MAX_QUESTIONS + 1)
    }

    payload = {
        "image": "photo of an object",
        "questions": too_many_questions,
    }
    res = client.post("/v1/systemone", json=payload)
    assert res.status_code == 400
    assert "exceeds the maximum allowed limit" in res.json()["detail"]


def test_rate_limiter_throttling_and_headers(monkeypatch):
    """Verify that sliding window rate limiter triggers HTTP 429 and returns proper headers."""
    import imagejev.server as srv
    client = TestClient(app)

    payload = {
        "image": "sample_image.png",
        "decision": "Quality check",
        "schema": {"type": "choice", "options": {"yes": "good", "no": "bad"}},
    }

    # Set DECIDE_RPM to 3 for this test
    monkeypatch.setattr(srv, "DECIDE_RPM", 3)
    rate_limiter.reset()

    # First 3 requests succeed
    for _ in range(3):
        r = client.post("/decide", json=payload)
        assert r.status_code == 200

    # 4th request must be throttled with HTTP 429
    r4 = client.post("/decide", json=payload)
    assert r4.status_code == 429
    assert "Retry-After" in r4.headers
    assert "X-RateLimit-Limit" in r4.headers
    assert "X-RateLimit-Remaining" in r4.headers
    assert r4.headers["X-RateLimit-Remaining"] == "0"
    data = r4.json()
    assert data["detail"]["error"] == "rate_limit_exceeded"


def test_api_key_auth_when_configured(monkeypatch):
    """Verify that API key enforcement works when keys are configured."""
    client = TestClient(app)
    VALID_API_KEYS.clear()
    VALID_API_KEYS.add("sk-secret-test-key")

    try:
        payload = {
            "image": "sample.jpg",
            "decision": "Check",
            "schema": {"type": "choice", "options": {"a": "a", "b": "b"}},
        }

        # Request without key -> 401
        res_no_key = client.post("/decide", json=payload)
        assert res_no_key.status_code == 401

        # Request with wrong key -> 401
        res_bad_key = client.post(
            "/decide",
            json=payload,
            headers={"X-API-Key": "sk-wrong-key"},
        )
        assert res_bad_key.status_code == 401

        # Request with valid key -> 200
        res_good_key = client.post(
            "/decide",
            json=payload,
            headers={"X-API-Key": "sk-secret-test-key"},
        )
        assert res_good_key.status_code == 200
    finally:
        VALID_API_KEYS.clear()


def test_protected_endpoint_auth_precedes_body_validation(monkeypatch):
    """Missing credentials must return 401 even when the body is malformed."""
    import imagejev.server as srv

    client = TestClient(app)
    monkeypatch.setattr(srv, "AUTH_REQUIRED", True)
    VALID_API_KEYS.clear()
    VALID_API_KEYS.add("sk-secret-test-key")

    try:
        response = client.post("/decide", json={})
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "ApiKey"
    finally:
        VALID_API_KEYS.clear()


def test_health_metrics_endpoint():
    """Verify health endpoint exposes limits and concurrency monitoring."""
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "concurrency" in data
    assert "rate_limiter" in data
    assert "limits" in data
    assert data["limits"]["max_batch_size"] == MAX_BATCH_SIZE


def test_ssrf_remote_url_blocked():
    """Verify that private IP / localhost remote image URLs are rejected for security."""
    from imagejev.backends import load_image_to_pil

    with pytest.raises(ValueError, match="forbidden|blocked"):
        load_image_to_pil("http://127.0.0.1:8000/secret.jpg")

    with pytest.raises(ValueError, match="forbidden|blocked"):
        load_image_to_pil("http://169.254.169.254/latest/meta-data")

    with pytest.raises(ValueError, match="forbidden|blocked"):
        load_image_to_pil("http://localhost:3000/private.png")
