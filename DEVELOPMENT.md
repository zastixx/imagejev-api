# Development and release source of truth

`imagejev-api-repo` is the canonical source tree for the published `imagejev`
package and HTTP service. The sibling `imagejev_sdk` directory is a historical
duplicate checkout and is not the source for new releases.

Use a clean virtual environment and install this repository in editable mode:

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[all]"
.\.venv\Scripts\python -m pytest
```

The active package should resolve to this repository's `src\imagejev` directory.
Do not rely on a global editable install or a manually set `PYTHONPATH`, because
the sibling historical checkout can otherwise shadow the intended package.

Before a release, build from this directory, inspect the wheel contents, install
the wheel in a clean environment, run the tests, and run `twine check --strict`.
Publishing and deployment require explicit approval and must use a reviewed
commit rather than an uncommitted working tree.
