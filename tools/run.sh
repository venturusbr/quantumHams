#!/usr/bin/env bash
# Build the quantumHams dev image and run it with the project root mounted
# into /workspace.
#
# Must be run from the project source root, e.g.:
#   tools/run.sh
#   tools/run.sh pytest
#   tools/run.sh bash
set -euo pipefail

IMAGE_NAME="quantumhams-dev"
DOCKERFILE="tools/Dockerfile"
PROJECT_ROOT="$(pwd)"

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "error: ${DOCKERFILE} not found; run this script from the project root (e.g. tools/run.sh)" >&2
  exit 1
fi

docker build --progress=quiet -t "${IMAGE_NAME}" -f "${DOCKERFILE}" tools

docker run --rm -it \
  -v "${PROJECT_ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE_NAME}" "$@"
