"""Unit tests for FastAPI REST server endpoints."""
import base64
import pytest

try:
    from fastapi.testclient import TestClient
    from imagejev.server import app, set_engine
    from imagejev import ImageJev, MockBackend
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_health_endpoint():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_decide_endpoint_choice():
    set_engine(ImageJev(MockBackend()))
    client = TestClient(app)

    payload = {
        "image": "photo of a cat",
        "decision": "Identify animal",
        "schema": {
            "type": "choice",
            "options": {"cat": "a cat", "dog": "a dog"}
        }
    }
    res = client.post("/decide", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "choice" in data
    assert "probability" in data
    assert "confidence" in data


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_decide_batch_endpoint():
    set_engine(ImageJev(MockBackend()))
    client = TestClient(app)

    payload = {
        "images": ["photo of a cat", "photo of a dog"],
        "decision": "Identify animal",
        "schema": {
            "type": "choice",
            "options": {"cat": "a cat", "dog": "a dog"}
        }
    }
    res = client.post("/decide_batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) == 2


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_systemone_multi_question_endpoint():
    set_engine(ImageJev(MockBackend()))
    client = TestClient(app)

    payload = {
        "image": "photo of a cat sitting on a mat",
        "questions": {
            "animal_type": {
                "type": "choice",
                "instructions": "Identify animal",
                "options": {"cat": "a cat", "dog": "a dog"}
            },
            "quality": {
                "type": "score",
                "instructions": "Rate image",
                "levels": ["poor", "good"]
            },
            "is_cat": {
                "type": "noul",
                "instructions": "Is it a cat?",
                "true": "a cat",
                "false": "not a cat"
            }
        }
    }
    res = client.post("/v1/systemone", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "answers" in data
    assert "animal_type" in data["answers"]
    assert "quality" in data["answers"]
    assert "is_cat" in data["answers"]
    assert data["answers"]["animal_type"]["type"] == "choice"
    assert data["answers"]["quality"]["type"] == "score"
    assert data["answers"]["is_cat"]["type"] == "noul"


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_decide_upload_endpoint():
    set_engine(ImageJev(MockBackend()))
    client = TestClient(app)

    # Test file upload with mock image bytes
    files = {"file": ("test.jpg", b"fake image bytes content", "image/jpeg")}
    data = {
        "decision": "Identify pet",
        "schema": '{"type": "choice", "options": {"dog": "a dog", "cat": "a cat"}}'
    }
    res = client.post("/decide_upload", files=files, data=data)
    assert res.status_code == 200
    res_json = res.json()
    assert "choice" in res_json
    assert "probability" in res_json
    assert "confidence" in res_json


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_decide_endpoint_invalid_schema():
    set_engine(ImageJev(MockBackend()))
    client = TestClient(app)

    payload = {
        "image": "some image",
        "decision": "test",
        "schema": {"type": "invalid_schema"}
    }
    res = client.post("/decide", json=payload)
    assert res.status_code == 422
