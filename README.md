## Getting started with uv (macOS via Homebrew)

Install uv
- Homebrew:
  ```bash
  brew install uv
  ```
- Verify:
  ```bash
  uv --version
  ```

Initialize a project and environment
- New project (or run in an existing repo):
  ```bash
  uv init .
  ```
- Create a virtual environment (uses your current Python; optionally specify a version):
  ```bash
  uv venv .venv
  # or
  uv venv --python 3.11 .venv
  ```
- You can activate the venv, or just use uv to run commands without activating:
  ```bash
  # optional activation (macOS/Linux)
  source .venv/bin/activate

  # preferred: run with uv
  uv run python -V
  uv run pytest
  ```

Add and manage dependencies
- Runtime deps:
  ```bash
  uv add flask
  ```
- Dev-only deps:
  ```bash
  uv add --dev pytest mypy
  ```
- Sync from project config (e.g., after cloning):
  ```bash
  uv sync
  ```
- Lock explicitly (if needed):
  ```bash
  uv lock
  ```

Where uv stores things
- Installed packages live inside your project’s virtual environment (e.g., `.venv/.../site-packages`).
- Download/build cache lives in `~/.cache/uv` on macOS. Useful commands:
  ```bash
  uv cache dir     # show cache path
  uv cache clean   # clear cache
  ```
  You can override the cache location with the `UV_CACHE_DIR` environment variable.

Files uv uses (reqs/Pipfile equivalents)
- Project dependencies are declared in `pyproject.toml`.
- Reproducible installs are captured in `uv.lock` (commit this to version control).
- Need a `requirements.txt` for tooling/CI? Generate it from your project:
  ```bash
  uv pip compile pyproject.toml -o requirements.txt
  ```

Notes
- If you use pyenv, `uv venv` will use the currently selected Python; you can also let uv manage a specific version via `--python`.
