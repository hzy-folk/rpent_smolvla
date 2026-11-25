# Copy to env.local.sh (ignored by Git), edit, then source it from the repo root.
export SMOLVLA_PYTHON="$PWD/.venv-smolvla/bin/python"
export RPENT_PYTHON="$PWD/.venv-rpent-libero-pro/bin/python"
export SMOLVLA_OSC7_CHECKPOINT="$PWD/checkpoints/smolvla/checkpoint_015000.pt"
export SMOLVLA_BASE_DIR="$PWD/checkpoints/smolvla/base"
export SMOLVLA_VLM_DIR="$PWD/checkpoints/smolvla/vlm"
export SAM3_CHECKPOINT_PATH="$PWD/checkpoints/sam3/sam3.pt"
# Optional overrides (the launcher already supplies these defaults):
# export RPENT_MEMORY_DIR="$PWD/robots/libero/smolvla_bridge/memory"
# export RPENT_AM_ENDPOINT='http://127.0.0.1:8320'
# export RPENT_OUTPUT_DIR="$PWD/logs/my_smolvla_run"
# Set the planner key in your shell; do not place credentials in tracked files.
# export OPENAI_BASE_URL='https://your-provider.example/v1'
# export RPENT_API_MODEL='openai-chat:your-vision-model'
