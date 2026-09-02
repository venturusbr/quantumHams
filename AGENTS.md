# AGENTS.md

Guidance for AI coding agents (Cline, Claude Code, etc.) working in this repository.

## Project overview

`quantumHams` (PyPI/import name `quantumhams`) is an extensible Python library for building
and time-evolving effective qubit **Hamiltonians** for quantum-computing hardware platforms.
It currently ships two physics models — rubidium Rydberg neutral-atom arrays and ytterbium
trapped-ion (Mølmer–Sørensen) chains — and two simulation backends — SciPy (CPU, always
available) and CUDA-Q (optional GPU, Linux x86-64 only).

Python ≥3.12, Alpha status, Apache-2.0 licensed. Author: Daniel de Haro Moraes. Maintainer:
Venturus.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
pre-commit install
```

Optional GPU path (only on Linux x86-64 with a working CUDA-Q install):

```bash
pip install -e ".[cudaq]"
```

Do not assume CUDA-Q is installed or installable in the agent's sandbox — treat
`tests/test_cudaq_backend.py` and the `cudaq_backend` module as environment-dependent; code
must keep working when CUDA-Q is absent (see `backends/auto_backend.py`'s fallback behavior).

## Commands

Run these from the repo root (hatch scripts are defined in `pyproject.toml`):

```bash
pytest                       # run tests (equivalent to `hatch run test`)
ruff check .                 # lint
ruff format --check .        # format check
ruff check --fix . && ruff format .   # auto-fix lint + format
mypy src/quantumhams         # strict type checking
pre-commit run --all-files   # everything pre-commit enforces (ruff + ruff-format)
python -m build               # build sdist/wheel
python -m twine check dist/*  # validate package metadata before release
mkdocs build --strict         # build docs, failing on warnings
```

Before considering a change complete, an agent should run at minimum `pytest`,
`ruff check .`, `ruff format --check .`, and `mypy src/quantumhams`. All are also run in CI
(`.github/workflows/ci.yml`) and as pre-commit hooks (`.pre-commit-config.yaml`, ruff only —
mypy/pytest are not pre-commit hooks, so they must be run manually).

### Test coverage requirement

`pytest` is configured (`pyproject.toml` → `[tool.pytest.ini_options]`) with
`--cov=quantumhams --cov-fail-under=90`. A PR/change that drops branch coverage below 90% will
fail CI — new code needs accompanying tests, not just a passing existing suite.

## Code style

- Formatting/linting via **ruff** (line length 100, target `py310`; rules: `E, F, I, B, UP,
  SIM, RUF`). Run `ruff format` rather than hand-formatting.
- **mypy --strict** on `src/quantumhams`. All new public functions/methods need full type
  annotations; the package ships `py.typed`, so type-correctness is part of its public
  contract.
- Prefer `numpy`/`scipy` idioms already used in the codebase (e.g. `scipy.sparse` Kronecker
  products in `core/pauli.py`) over introducing new dependencies. Only `numpy>=1.26` and
  `scipy>=1.12` are core runtime dependencies; `cudaq` is optional/extra. Don't add new hard
  dependencies without strong justification.

## Architecture (read before modifying)

```
src/quantumhams/
  core/
    model.py     — HamiltonianModel ABC (n_qubits, pauli_hamiltonian() -> PauliHamiltonian;
                    matrix() and evolve() come for free)
    pauli.py     — PauliTerm / PauliHamiltonian (sparse-matrix construction)
    states.py    — basis-state helpers ("01" or int) and state-vector normalization
    results.py   — EvolutionResult dataclass (times, states, backend, .probabilities)
  backends/
    base.py      — SimulationBackend Protocol: evolve(hamiltonian, initial_state, times)
    scipy_backend.py  — CPU reference backend (scipy.sparse.linalg.expm_multiply)
    cudaq_backend.py  — optional GPU backend via NVIDIA CUDA-Q
    auto_backend.py   — tries cudaq, falls back to scipy with RuntimeWarning on
                         ImportError/RuntimeError/OSError
    registry.py  — get_backend("auto"|"scipy"|"cudaq") and backend_status()
  neutral_atoms/rubidium.py   — NeutralAtomRegister, RubidiumRydbergHamiltonian
  trapped_ions/ytterbium.py   — YtterbiumIonChain, YtterbiumMolmerSorensenHamiltonian
```

**The seam to protect:** physics models only ever produce a backend-neutral
`PauliHamiltonian`; backends only ever consume `PauliHamiltonian` + initial state + times.
Never let a `HamiltonianModel` subclass import from `backends/`, and never let a backend
depend on a specific physics model (`neutral_atoms`/`trapped_ions`). This is what keeps new
hardware technologies and new accelerators independently pluggable — preserve it in every
change.

CUDA-Q qubit ordering is reversed relative to quantumHams' own convention; this is handled
deliberately in `cudaq_backend.py` when converting the Pauli decomposition. Don't "simplify"
that reversal without re-verifying against `tests/test_cudaq_backend.py`.

## Adding a new Hamiltonian model (physics)

Per `docs/extending.md`:

1. Create a technology-specific package under `src/quantumhams/`.
2. Implement a class derived from `HamiltonianModel`.
3. Expose `n_qubits` and `pauli_hamiltonian()`, expressed with `PauliTerm` objects.
4. Add analytical unit tests for small systems and numerical tests for invariants (e.g.
   Hermiticity, expected steady states) — follow the style of `tests/test_neutral_atoms.py`
   / `tests/test_trapped_ions.py`.
5. Export the stable public class from `src/quantumhams/__init__.py`.
6. Document physical assumptions and units in the class docstring — this project's convention
   (see `CONTRIBUTING.md`) is that every physics model documents its assumptions/units
   explicitly, since Hamiltonian coefficients are physically meaningful, not arbitrary.

## Adding a new backend

Implement the `SimulationBackend` Protocol (`backends/base.py`) and register it in
`backends/registry.py`. Add tests mirroring `tests/test_backends.py`. If the backend has an
optional/heavy dependency (as CUDA-Q does), guard the import and make `auto_backend.py`-style
fallback behavior explicit and tested.

## Documentation

- `docs/` is built with MkDocs + mkdocstrings (`mkdocs.yml`) and deployed via
  `.github/workflows/docs.yml`. API docs are largely generated from docstrings — keep
  docstrings accurate and complete for anything public.
- Update `docs/usage.md` / `docs/extending.md` and `README.md` when public API or usage
  patterns change; run `mkdocs build --strict` to catch broken references before finishing.

## Commits / PRs

- Use feature branches and conventional, focused commits (per `CONTRIBUTING.md`).
- Public API changes require a `CHANGELOG.md` entry.
- Before opening/finishing a PR-sized change, run the full sequence from `CONTRIBUTING.md`:
  `pre-commit run --all-files`, `pytest`, `python -m build`, `python -m twine check dist/*`,
  `mkdocs build --strict`.
- Don't bump the version in `pyproject.toml` unless explicitly asked to cut a release.

## Things to avoid

- Don't weaken `mypy --strict` compliance (e.g. adding `# type: ignore` without a real reason).
- Don't drop overall branch coverage below 90%.
- Don't couple `core/` or `backends/` to a specific physics model, or couple a physics model to
  a specific backend.
- Don't assume a GPU or CUDA-Q is present; any code path exercised by default tests must work
  CPU-only via the SciPy backend.
