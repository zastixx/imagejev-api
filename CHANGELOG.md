# Changelog

All notable changes to `imagejev` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-30

### Added
- Core `ImageJev` class with `decide` and `decide_batch` methods.
- Support for `choice`, `score` (with `dispersion`), and `noul` schemas matching standard System-One specifications.
- Production `OpenCLIPBackend` with support for CUDA, MPS, and CPU.
- `MockBackend` for lightweight, deterministic offline unit testing.
- Support for image filepaths (`str` / `Path`), `PIL.Image`, raw `bytes`, and remote `http://` / `https://` URLs.
- Strongly typed result dataclasses (`ChoiceResult`, `ScoreResult`, `NoulResult`) with backward-compatible dict behavior.
- FastAPI HTTP server and CLI `imagejev-server` with `/decide`, `/decide_batch`, and `/health` endpoints.
- Heuristic warnings for bare judgment words (`yes/no`) preventing degraded CLIP prompt construction.
- Full packaging setup with `pyproject.toml`, type marker `py.typed`, and multi-OS CI/CD workflows.
