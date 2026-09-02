#!/usr/bin/env bash
# Container entrypoint for the quantumHams dev image.
#
# Runs on every `docker run`, after tools/run.sh has bind-mounted the project
# root into /workspace. Installs quantumhams in editable/dev mode against
# whatever source is currently on the mount, so edits made on the host are
# reflected immediately inside the container without rebuilding the image.
set -euo pipefail

pip install --quiet -e ".[dev,docs]"

exec "$@"
