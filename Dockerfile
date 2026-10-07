# Multi-stage production Dockerfile for ImageJev Decision Server
# Supports NVIDIA CUDA GPU acceleration and CPU fallback

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and source
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install imagejev and server dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build && \
    pip install --no-cache-dir ".[server]"

# ---------------------------------------------------------------------------------
# Production Runtime Image
# ---------------------------------------------------------------------------------
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime system dependencies for PIL / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Pre-cache OpenCLIP model weights at build time (ViT-B-32 / laion2b_s34b_b79k)
RUN python -c "from imagejev import ImageJev; ImageJev()"

# Expose FastAPI REST Port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Start production server with single worker (safeguards 2GB RAM instances)
CMD ["sh", "-c", "uvicorn imagejev.server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
