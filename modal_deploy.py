"""Deploy the ImageJev FastAPI service on Modal.

Deploy from this repository with::

    modal deploy modal_deploy.py

The app name and web-function name intentionally match the existing public
deployment, so a successful redeploy keeps this URL stable::

    https://rajsubhu169--imagejev-api-fastapi-app.modal.run
"""

import os
import sys

import modal


APP_NAME = "imagejev-api"


image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime",
        add_python="3.11",
    )
    .pip_install(
        "open-clip-torch>=2.20.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.6",
        "pillow>=9.0.0",
        "numpy>=1.22.0",
        "requests>=2.28.0",
    )
    .add_local_dir("src", remote_path="/app/src", copy=True)
    .add_local_file("pyproject.toml", remote_path="/app/pyproject.toml", copy=True)
    .add_local_file("README.md", remote_path="/app/README.md", copy=True)
    .add_local_file("LICENSE", remote_path="/app/LICENSE", copy=True)
    .run_commands("pip install /app")
    .run_commands(
        "python -c 'from imagejev import ImageJev; ImageJev(); print(\"CLIP ViT-B-32 weights cached OK\")'"
    )
)

app = modal.App(
    APP_NAME,
    image=image,
    secrets=[modal.Secret.from_name("imagejev-api-secrets")],
)


@app.function(
    cpu=2,
    memory=4096,
    min_containers=1,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def fastapi_app():
    """Return the ImageJev FastAPI application."""
    sys.path.insert(0, "/app/src")

    os.environ.setdefault("IMAGEJEV_ENV", "production")
    os.environ.setdefault("IMAGEJEV_PRELOAD", "true")
    os.environ.setdefault("IMAGEJEV_RATE_LIMIT_ENABLED", "true")
    os.environ.setdefault("IMAGEJEV_RATE_LIMIT_PER_MINUTE", "60")
    os.environ.setdefault("IMAGEJEV_BATCH_RATE_LIMIT_PER_MINUTE", "15")
    os.environ.setdefault("IMAGEJEV_MAX_BATCH_SIZE", "50")
    os.environ.setdefault("IMAGEJEV_MAX_QUESTIONS", "25")
    os.environ.setdefault("IMAGEJEV_MAX_CONCURRENCY", "4")
    os.environ.setdefault("PORT", "8000")

    from imagejev.server import app as imagejev_fastapi_app

    return imagejev_fastapi_app
