# Changelog

All notable changes to `imagejev` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-10-15

- Include the remote `ImageJevClient` and typed HTTP errors in published distributions.
- Align package and API version reporting with the release version.
- Improve release validation, import provenance, and deterministic CI coverage.

## [0.1.1] - 2026-10-11

- Replaced the PyPI long description with evidence-qualified documentation.
- Removed unsupported performance, cost, and correctness claims.
- Clarified benchmark scope, limitations, and server defaults.
- Added the synchronous `ImageJevClient` with typed result parsing and HTTP errors.
- Added `modal_deploy.py` for the public FastAPI deployment.

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
