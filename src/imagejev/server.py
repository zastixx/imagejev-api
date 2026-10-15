"""FastAPI HTTP Server and CLI for ImageJev.

Production-grade server featuring:
- Sliding-window per-IP / per-client rate limiting (no external redis/slowapi dependency required).
- Concurrency bounding gate (prevents CPU thread thrashing and OOM spikes).
- Strict batch size caps on /decide_batch.
- Maximum question count cap on /v1/systemone.
- Maximum image payload/upload size validation (HTTP 413).
- Optional API key authentication via X-API-Key header.
- CORS configuration and production health monitoring.

Run with:
    imagejev-server --host 0.0.0.0 --port 8000
    or:
    uvicorn imagejev.server:app --reload
"""
from __future__ import annotations

import base64
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .backends import MockBackend
from .core import ImageJev
from .rate_limiter import ConcurrencyGate, SlidingWindowRateLimiter, get_client_ip
from . import __version__

try:
    from fastapi import FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict, Field

    # ── Configuration & Environment Variables ─────────────────────────────────────

    # API Keys: If empty/unset, auth is disabled (dev mode). If set, X-API-Key is enforced.
    _RAW_API_KEYS = os.getenv("IMAGEJEV_API_KEYS", "").strip()
    VALID_API_KEYS: set[str] = set(k.strip() for k in _RAW_API_KEYS.split(",") if k.strip())

    # Rate Limiting configuration
    RATE_LIMIT_ENABLED = os.getenv("IMAGEJEV_RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes")
    DECIDE_RPM = int(os.getenv("IMAGEJEV_RATE_LIMIT_PER_MINUTE", "60"))
    BATCH_RPM = int(os.getenv("IMAGEJEV_BATCH_RATE_LIMIT_PER_MINUTE", "15"))

    # Size and batch caps
    MAX_BATCH_SIZE = int(os.getenv("IMAGEJEV_MAX_BATCH_SIZE", "50"))
    MAX_QUESTIONS = int(os.getenv("IMAGEJEV_MAX_QUESTIONS", "25"))
    MAX_IMAGE_BYTES = int(os.getenv("IMAGEJEV_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))  # 10MB default
    MAX_CONCURRENCY = int(os.getenv("IMAGEJEV_MAX_CONCURRENCY", "4"))
    CONCURRENCY_TIMEOUT = float(os.getenv("IMAGEJEV_CONCURRENCY_TIMEOUT", "30.0"))
    TRUST_PROXY_HEADERS = os.getenv("IMAGEJEV_TRUST_PROXY_HEADERS", "false").lower() in ("true", "1", "yes")

    # ── Protection Primitives ─────────────────────────────────────────────────────

    rate_limiter = SlidingWindowRateLimiter(
        requests_per_minute=DECIDE_RPM,
        window_seconds=60.0,
        enabled=RATE_LIMIT_ENABLED,
    )

    concurrency_gate = ConcurrencyGate(
        max_concurrent=MAX_CONCURRENCY,
        timeout=CONCURRENCY_TIMEOUT,
    )

    # ── Pydantic Request Models ───────────────────────────────────────────────────

    class DecideRequest(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        image: str = Field(description="Base64 encoded image string, local filepath, or public HTTP/HTTPS URL")
        decision: str = Field(description="Plain-language decision question or statement")
        schema_: dict = Field(default_factory=dict, alias="schema", description="Choice, score, or noul schema")

    class DecideBatchRequest(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        images: List[str] = Field(description="List of Base64 strings, filepaths, or URLs")
        decision: str = Field(description="Plain-language decision question or statement")
        schema_: dict = Field(default_factory=dict, alias="schema", description="Choice, score, or noul schema")

    class SystemOneRequest(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        image: str = Field(description="Base64 encoded image string, local filepath, or public HTTP/HTTPS URL")
        questions: Dict[str, dict] = Field(description="Mapping of question_id to schema definition")

    def _validate_schema(schema: dict) -> None:
        if not isinstance(schema, dict):
            raise ValueError("schema must be an object")
        schema_type = schema.get("type")
        if schema_type == "choice":
            options = schema.get("options")
            if not isinstance(options, (dict, list, tuple)) or len(options) < 2:
                raise ValueError("choice schema requires at least two options")
        elif schema_type == "score":
            levels = schema.get("levels")
            if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
                raise ValueError("score schema requires 2-10 levels")
        elif schema_type == "noul":
            if schema.get("true") is not None and not isinstance(schema.get("true"), str):
                raise ValueError("noul schema 'true' must be a string")
            if schema.get("false") is not None and not isinstance(schema.get("false"), str):
                raise ValueError("noul schema 'false' must be a string")
        else:
            raise ValueError('schema["type"] must be "choice", "score", or "noul"')

    from contextlib import asynccontextmanager

    _engine_instance: Dict[str, ImageJev] = {}
    _engine_lock = threading.Lock()

    def get_engine() -> ImageJev:
        if "engine" not in _engine_instance:
            with _engine_lock:
                if "engine" not in _engine_instance:
                    _engine_instance["engine"] = ImageJev()
        return _engine_instance["engine"]

    def set_engine(engine: ImageJev) -> None:
        _engine_instance["engine"] = engine

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        """Preload engine weights during server startup if IMAGEJEV_PRELOAD is enabled."""
        if os.getenv("IMAGEJEV_PRELOAD", "false").lower() in ("true", "1", "yes"):
            get_engine()
        yield

    # ── Application Setup ─────────────────────────────────────────────────────────

    app = FastAPI(
        title="ImageJev API",
        version=__version__,
        description=(
            "Image-native structured decision engine: IMAGE + DECISION + SCHEMA -> structured decision.\n\n"
            "Features built-in per-IP rate limiting, batch size validation, and concurrency throttling."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    _raw_origins = os.getenv("IMAGEJEV_ALLOWED_ORIGINS", "").strip()
    _allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=bool(_allowed_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # ── Security & Rate Limit Helpers ─────────────────────────────────────────────

    def _verify_api_key_or_raise(x_api_key: Optional[str]) -> None:
        """Enforce API key validation if VALID_API_KEYS is configured."""
        if not VALID_API_KEYS:
            return  # Open access / dev mode
        if not x_api_key or x_api_key.strip() not in VALID_API_KEYS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized. A valid 'X-API-Key' header is required.",
            )

    def _enforce_rate_limit(request: Request, limit: int, cost: int = 1) -> None:
        """Check rate limit for client IP / key and raise HTTP 429 if exceeded."""
        if not rate_limiter.enabled:
            return

        headers_dict = {k.lower(): v for k, v in request.headers.items()}
        client_host = request.client.host if request.client else "unknown"
        # If API key is present, rate limit by key, otherwise by IP
        api_key = headers_dict.get("x-api-key")
        client_ip = get_client_ip(headers_dict, client_host) if TRUST_PROXY_HEADERS else client_host
        client_key = f"key:{api_key.strip()}" if api_key and VALID_API_KEYS else f"ip:{client_ip}"

        allowed, remaining, retry_after, reset_in = rate_limiter.check(
            client_key=client_key,
            limit=limit,
            cost=cost,
        )

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"Too many requests. Limit is {limit} req/min. Please retry after {retry_after}s.",
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_in),
                },
            )

    def _decode_image_input(img_str: str) -> Any:
        s = img_str.strip()
        # Data URI format
        if s.startswith("data:image/") and ";base64," in s:
            s = s.split(";base64,")[1]
            raw = base64.b64decode(s)
            if len(raw) > MAX_IMAGE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Image data URI payload ({len(raw)} bytes) exceeds maximum allowed size ({MAX_IMAGE_BYTES} bytes).",
                )
            return raw

        # URL
        if s.startswith(("http://", "https://")):
            return s

        # Local existing file path
        if (len(s) < 260) and (Path(s).is_file() if "\n" not in s else False):
            return s

        # Base64 decode attempt
        try:
            decoded = base64.b64decode(s)
            if len(decoded) > 0:
                if len(decoded) > MAX_IMAGE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Image base64 payload ({len(decoded)} bytes) exceeds maximum allowed size ({MAX_IMAGE_BYTES} bytes).",
                    )
                return decoded
        except HTTPException:
            raise
        except Exception:
            pass

        return s

    # ── Endpoints ─────────────────────────────────────────────────────────────────

    @app.get("/health", status_code=status.HTTP_200_OK)
    def health_check():
        """Public system health check and throughput metrics."""
        return {
            "status": "ok",
            "service": "imagejev",
            "version": "0.2.0",
            "concurrency": {
                "in_flight": concurrency_gate.in_flight,
                "max_concurrent": concurrency_gate.max_concurrent,
            },
            "rate_limiter": {
                "enabled": rate_limiter.enabled,
                "default_rpm": rate_limiter.default_limit,
            },
            "limits": {
                "max_batch_size": MAX_BATCH_SIZE,
                "max_questions": MAX_QUESTIONS,
                "max_image_mb": MAX_IMAGE_BYTES // (1024 * 1024),
            },
        }

    @app.post("/decide", response_model=Dict[str, Any])
    def decide_endpoint(
        req: DecideRequest,
        request: Request,
        x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    ):
        """Make a structured decision on a single visual input."""
        _verify_api_key_or_raise(x_api_key)
        _enforce_rate_limit(request, limit=DECIDE_RPM, cost=1)

        if not concurrency_gate.acquire():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server is at maximum processing capacity. Please retry momentarily.",
            )

        try:
            engine = get_engine()
            image = _decode_image_input(req.image)
            _validate_schema(req.schema_)
            return engine.decide(image, req.decision, req.schema_)
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e))
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=500, detail="Internal decision error")
        finally:
            concurrency_gate.release()

    @app.post("/decide_upload", response_model=Dict[str, Any])
    async def decide_upload_endpoint(
        request: Request,
        file: UploadFile = File(..., description="Local image file to upload (JPEG, PNG, WebP)"),
        decision: str = Form("Is this product suitable for listing?", description="Decision question or prompt"),
        schema_str: str = Form(
            '{"type": "choice", "options": {"yes": "a clear high quality photo", "no": "a blurry or dark photo"}}',
            alias="schema",
            description="JSON schema string",
        ),
        x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    ):
        """Upload a local image file directly from client machine or browser."""
        _verify_api_key_or_raise(x_api_key)
        _enforce_rate_limit(request, limit=DECIDE_RPM, cost=1)

        contents = await file.read()
        if len(contents) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded file size ({len(contents)} bytes) exceeds limit of {MAX_IMAGE_BYTES} bytes ({MAX_IMAGE_BYTES // (1024 * 1024)}MB).",
            )

        if not concurrency_gate.acquire():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server is at maximum processing capacity. Please retry momentarily.",
            )

        try:
            schema_dict = json.loads(schema_str)
            _validate_schema(schema_dict)
            engine = get_engine()
            return engine.decide(contents, decision, schema_dict)
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="Invalid JSON in schema form field")
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e))
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=500, detail="Internal decision error")
        finally:
            concurrency_gate.release()

    @app.post("/decide_batch", response_model=List[Dict[str, Any]])
    def decide_batch_endpoint(
        req: DecideBatchRequest,
        request: Request,
        x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    ):
        """Batch decision processing with strict batch size caps and rate limiting."""
        _verify_api_key_or_raise(x_api_key)

        # Batch size validation
        if not req.images:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Images array cannot be empty.",
            )

        if len(req.images) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size ({len(req.images)}) exceeds the maximum allowed limit of {MAX_BATCH_SIZE} images per request.",
            )

        _enforce_rate_limit(request, limit=BATCH_RPM, cost=1)

        if not concurrency_gate.acquire():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server is at maximum processing capacity. Please retry momentarily.",
            )

        try:
            engine = get_engine()
            decoded_images = [_decode_image_input(img) for img in req.images]
            _validate_schema(req.schema_)
            return engine.decide_batch(decoded_images, req.decision, req.schema_)
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e))
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=500, detail="Internal decision error")
        finally:
            concurrency_gate.release()

    @app.post("/v1/systemone", response_model=Dict[str, Any])
    @app.post("/v1/decide", response_model=Dict[str, Any])
    def systemone_endpoint(
        req: SystemOneRequest,
        request: Request,
        x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    ):
        """Multi-question structured decision endpoint on a single forward pass."""
        _verify_api_key_or_raise(x_api_key)

        # Question count validation
        if not req.questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Questions mapping cannot be empty.",
            )

        if len(req.questions) > MAX_QUESTIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Questions count ({len(req.questions)}) exceeds the maximum allowed limit of {MAX_QUESTIONS} per request.",
            )

        _enforce_rate_limit(request, limit=DECIDE_RPM, cost=1)

        if not concurrency_gate.acquire():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server is at maximum processing capacity. Please retry momentarily.",
            )

        try:
            engine = get_engine()
            image = _decode_image_input(req.image)
            for question in req.questions.values():
                _validate_schema(question.get("schema", question))
            answers = engine.decide_multi(image, req.questions)
            return {"answers": answers}
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e))
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=500, detail="Internal decision error")
        finally:
            concurrency_gate.release()

    def main():
        import argparse
        import uvicorn

        parser = argparse.ArgumentParser(description="ImageJev HTTP Server")
        parser.add_argument("--host", default="0.0.0.0", help="Bind host")
        parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")), help="Bind port")
        parser.add_argument("--reload", action="store_true", help="Enable live auto-reload")
        parser.add_argument("--mock", action="store_true", help="Use mock backend for testing")
        args = parser.parse_args()

        if args.mock:
            set_engine(ImageJev(MockBackend()))

        uvicorn.run("imagejev.server:app", host=args.host, port=args.port, reload=args.reload)

except ImportError:
    app = None  # type: ignore[no-redef, assignment]

    def main():
        print("FastAPI / Uvicorn are not installed. Install with: pip install imagejev[server]")
