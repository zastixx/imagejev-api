"""Production server entrypoint for ImageJev.

Aliases directly into imagejev.server to provide full production protection:
- Sliding window per-IP rate limiting
- Concurrency bounding gate
- Batch size caps & Question count caps
- Image payload size caps
- Optional API key authentication
"""
from .server import app, main, get_engine, set_engine

__all__ = ["app", "main", "get_engine", "set_engine"]

if __name__ == "__main__":
    main()
