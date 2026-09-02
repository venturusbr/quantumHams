#!/usr/bin/env bash
# Runs every check listed in the "Commands" section of AGENTS.md, in one pass.
#
# Intended to be run inside the containerized dev environment:
#   tools/run.sh tools/checks.sh
#
# (also works directly on a local venv: tools/checks.sh)
#
# Each check runs to completion even if an earlier one fails, so a single
# invocation reports the full picture instead of stopping at the first
# failure. Prints a pass/fail summary at the end and exits non-zero if any
# check failed.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

failures=()

run_check() {
  local name="$1"
  shift
  echo
  echo "==> ${name}"
  if "$@"; then
    echo "==> ${name}: OK"
  else
    echo "==> ${name}: FAILED"
    failures+=("${name}")
  fi
}

run_check "pytest" pytest
run_check "ruff check ." ruff check .
run_check "ruff format --check ." ruff format --check .
run_check "mypy src/quantumhams" mypy src/quantumhams
# FIXME: git does not recognise /workspace as a git directory
# run_check "pre-commit run --all-files" pre-commit run --all-files
run_check "python -m build" python -m build
run_check "twine check dist/*" bash -c 'python -m twine check dist/*'
run_check "mkdocs build --strict" mkdocs build --strict

echo
if [[ "${#failures[@]}" -eq 0 ]]; then
  echo "All checks passed."
  exit 0
else
  echo "Failed checks:"
  for name in "${failures[@]}"; do
    echo "  - ${name}"
  done
  exit 1
fi
