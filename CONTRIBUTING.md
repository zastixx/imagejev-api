# Contributing to ImageJev

Thank you for your interest in contributing to `imagejev`!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/imagejev.git
   cd imagejev
   ```

2. Create a virtual environment and install development dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[all]"
   ```

3. Run the test suite:
   ```bash
   pytest tests/
   ```

## Code Style & Standards

- Follow PEP 8 guidelines.
- Use type annotations everywhere (`mypy` compliant).
- All new features or primitives must include unit tests in `tests/`.
- Ensure PRs pass all CI checks before requesting review.

## Submitting Pull Requests

1. Fork the repo and create your branch from `main`.
2. Ensure existing tests pass: `pytest`.
3. Open a Pull Request with a clear description of changes.
