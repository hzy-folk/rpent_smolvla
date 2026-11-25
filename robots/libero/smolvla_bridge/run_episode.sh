#!/usr/bin/env bash
# Run the salad-dressing seed-0 episode against an already running bridge.
set -euo pipefail
BRIDGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$BRIDGE_DIR/../../.." && pwd)"
cd "$REPO_DIR"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in your shell}"
: "${RPENT_API_MODEL:?Set RPENT_API_MODEL, including the openai-chat: prefix}"
: "${SAM3_CHECKPOINT_PATH:?Set SAM3_CHECKPOINT_PATH to your SAM3 weights}"
exec "${RPENT_PYTHON:-$REPO_DIR/.venv-rpent-libero-pro/bin/python}" \
  -m rpent.cli.main \
  --robot libero --libero-type pro \
  --suite libero_object_task --task 8 --seed 0 \
  --planner api --model "$RPENT_API_MODEL" --reasoning-effort low \
  --memory-profile local --memory-dir "${RPENT_MEMORY_DIR:-$BRIDGE_DIR/memory}" \
  --am-backend pi05 --am-endpoint "${RPENT_AM_ENDPOINT:-http://127.0.0.1:8320}" \
  --max-turns 60 --record-trajectory \
  --output-dir "${RPENT_OUTPUT_DIR:-logs/smolvla_seed0_$(date +%Y%m%d_%H%M%S)}" \
  "$@"
