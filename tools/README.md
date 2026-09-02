# tools

Containerized development environment for quantumHams. Lets you build, test, and run the
project inside a disposable Docker container instead of a local virtualenv.

## Files

- **`Dockerfile`** — minimal `python:3.12-slim` image (matches `requires-python = ">=3.12"` in
  `pyproject.toml`) with `git` installed via `apt-get` (needed by `pre-commit` and general dev
  workflows). It does **not** copy the project source into the image; `run.sh` bind-mounts
  the project root into `/workspace` at container start instead. Rebuilding the image is only
  needed when `Dockerfile` or `entrypoint.sh` itself changes.
- **`entrypoint.sh`** — the image's `ENTRYPOINT`. On every `docker run`, after the project root
  is mounted at `/workspace`, it runs `pip install --quiet -e ".[dev,docs]"` against the
  mounted source and then `exec`s whatever command was passed to the container. This installs
  quantumhams in editable/dev mode fresh each start, so edits made on the host under
  `src/quantumhams/` are picked up immediately without rebuilding the image.
- **`run.sh`** — builds the `quantumhams-dev` image from `Dockerfile` and runs it with the
  project root mounted into `/workspace`. Must be invoked from the project root.
- **`checks.sh`** — runs every check listed in the "Commands" section of `AGENTS.md` (`pytest`,
  `ruff check`, `ruff format --check`, `mypy`, `pre-commit run --all-files`, `python -m build`,
  `twine check dist/*`, `mkdocs build --strict`) in one pass. Each check runs to completion even
  if an earlier one failed, so a single invocation reports the full picture; it prints a
  pass/fail summary at the end and exits non-zero if anything failed.

## Usage

From the project root:

```bash
tools/run.sh            # build the image, install quantumhams editable, drop into bash
tools/run.sh pytest     # same, then run the test suite instead of an interactive shell
tools/run.sh tools/checks.sh   # run every AGENTS.md check in the container, in one go
```

Any command works — everything after `tools/run.sh` is passed straight to the container.
`checks.sh` also works outside the container, against a local venv with `.[dev,docs]`
installed (`tools/checks.sh`).

## Notes / trade-offs

- Running from the wrong directory fails fast: `run.sh` checks for `tools/Dockerfile` relative
  to the current working directory and errors out if it's not found.
- Because the editable install runs at container *start* (not image *build*) time, every
  invocation of `run.sh` re-installs `.[dev,docs]` over the network unless pip's cache is warm.
  This keeps the setup simple (no build-context/dependency-prefetch tricks) at the cost of a
  slightly slower startup — an accepted trade-off; see `AGENTS.md` history/PR discussion if this
  ever needs to be optimized.
- CUDA-Q (the optional `cudaq` extra) is not installed in this image — it is Linux x86-64-only
  and not required for the SciPy backend that CI and this container exercise.
