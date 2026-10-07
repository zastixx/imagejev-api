"""Vision and Text Encoders for ImageJev.

Provides production OpenCLIP backend and lightweight MockBackend for testing.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path
from typing import Any, Protocol, Sequence, Union

import numpy as np


class Backend(Protocol):
    """Protocol for ImageJev feature encoders."""

    logit_scale: float

    def encode_image(self, image: Any) -> np.ndarray:
        """Encode a single image into an L2-normalized 1D embedding vector of shape (D,)."""
        ...

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        """Encode a batch of images into L2-normalized 2D matrix of shape (B, D)."""
        ...

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a sequence of text strings into L2-normalized matrix of shape (N, D)."""
        ...


def load_image_to_pil(image: Any) -> Any:
    """Load various image inputs into a standard RGB PIL Image.

    Supports:
        - PIL.Image.Image objects
        - Local file paths (str or pathlib.Path)
        - Raw image bytes
        - io.BytesIO or file-like streams
        - Remote URLs (http:// or https://) with SSRF & size protections
    """
    import ipaddress
    import urllib.parse
    from PIL import Image

    # Set decompression bomb threshold to 25 megapixels to prevent memory exhaustion DoS
    Image.MAX_IMAGE_PIXELS = 25_000_000

    if isinstance(image, (str, Path)):
        path_or_url = str(image).strip()
        if path_or_url.startswith(("http://", "https://")):
            parsed = urllib.parse.urlparse(path_or_url)
            hostname = parsed.hostname
            if not hostname:
                raise ValueError(f"Invalid remote image URL: {path_or_url}")

            # SSRF Protection: block private and loopback networks
            try:
                ip = ipaddress.ip_address(hostname)
                is_ip = True
            except ValueError:
                is_ip = False

            if is_ip:
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                    raise ValueError(f"Access to private/local network address {hostname} is forbidden.")
            else:
                # String hostname: block known internal / cloud metadata hostnames
                blocked_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "metadata.google.internal"}
                if hostname.lower() in blocked_hosts or hostname.lower().endswith(".internal") or hostname.lower().endswith(".local"):
                    raise ValueError(f"Access to hostname {hostname} is forbidden.")

            req = urllib.request.Request(
                path_or_url,
                headers={"User-Agent": "ImageJev-SDK/0.2.0"}
            )
            # Read in chunks with a hard 10MB limit to prevent memory exhaustion
            max_bytes = 10 * 1024 * 1024
            chunks = []
            total_read = 0
            with urllib.request.urlopen(req, timeout=12) as resp:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total_read += len(chunk)
                    if total_read > max_bytes:
                        raise ValueError(f"Remote image exceeds maximum allowed size of {max_bytes // (1024*1024)}MB.")
            data = b"".join(chunks)
            return Image.open(io.BytesIO(data)).convert("RGB")

        return Image.open(path_or_url).convert("RGB")

    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")

    if hasattr(image, "read") and callable(image.read):
        return Image.open(image).convert("RGB")

    if hasattr(image, "convert"):
        return image.convert("RGB")

    raise TypeError(
        f"Unsupported image input type: {type(image)!r}. "
        f"Expected filepath (str/Path), URL (http/https), PIL.Image, or raw bytes."
    )


class OpenCLIPBackend:
    """Production vision & text encoder powered by OpenCLIP.

    Default backbone: ViT-B-32 trained on LAION-2B (laion2b_s34b_b79k).
    """

    def __init__(
        self,
        name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        import open_clip
        import torch

        self.torch = torch
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            name,
            pretrained=pretrained,
            device=self.device,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        self.model.eval()
        self.tok = open_clip.get_tokenizer(name)
        self.logit_scale = float(self.model.logit_scale.exp().detach().cpu().item())

    def encode_image(self, image: Any) -> np.ndarray:
        pil_img = load_image_to_pil(image)
        with self.torch.inference_mode():
            tensor = self.pre(pil_img).unsqueeze(0).to(self.device)
            v = self.model.encode_image(tensor)[0]
            v = (v / v.norm()).float().cpu().numpy()
            return v

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        pil_images = [load_image_to_pil(img) for img in images]
        with self.torch.inference_mode():
            tensors = self.torch.stack([self.pre(img) for img in pil_images]).to(self.device)
            v = self.model.encode_image(tensors)
            v = (v / v.norm(dim=-1, keepdim=True)).float().cpu().numpy()
            return v

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        with self.torch.inference_mode():
            tokens = self.tok(list(texts)).to(self.device)
            v = self.model.encode_text(tokens)
            v = (v / v.norm(dim=-1, keepdim=True)).float().cpu().numpy()
            return v


class MockBackend:
    """Deterministic hash-based backend for fast, network-free unit tests."""

    logit_scale: float = 100.0

    def __init__(self, dim: int = 256, noise: float = 0.0, seed: int = 0):
        import hashlib

        self.dim = dim
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self._hashlib = hashlib

    def _tok(self, w: str) -> np.ndarray:
        seed = int(self._hashlib.md5(w.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).standard_normal(self.dim)

    def _vec(self, text: str) -> np.ndarray:
        words = text.lower().replace(":", " ").split() or [""]
        v = sum(self._tok(w) for w in words)
        return (v / max(np.linalg.norm(v), 1e-9)).astype(np.float32)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_image(self, image: Any) -> np.ndarray:
        text = image if isinstance(image, str) else str(image)
        v = self._vec(text) + self.noise * self.rng.standard_normal(self.dim) / np.sqrt(self.dim)
        return (v / max(np.linalg.norm(v), 1e-9)).astype(np.float32)

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        return np.stack([self.encode_image(img) for img in images]).astype(np.float32)
