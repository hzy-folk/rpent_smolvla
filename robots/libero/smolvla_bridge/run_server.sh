#!/usr/bin/env bash
# Run independently of the caller's working directory. No RPent install required.
set -euo pipefail
BRIDGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SMOLVLA_PYTHON:-${PYTHON:-python3}}" "$BRIDGE_DIR/server.py" "$@"
