# RPent + SmolVLA: LIBERO-Pro salad dressing

[简体中文](README.md)

This directory contains the standalone `vla.predict` HTTP bridge, portable profiles, launch scripts, and a small reference memory corpus. It needs no separate bridge checkout. The API planner localizes objects and selects tools; SmolVLA supplies grasp action chunks; RPent primitives move the arm and release the object.

The example uses LIBERO-Pro, `libero_object_task`, task 8, seed 0: put the salad dressing into the basket. Historical runs include both success and failure on seed 0. This is a runnable integration procedure, not a claim of repeatable task success.

## Install two environments

Use Linux, an NVIDIA GPU, and EGL. The development machine has an 8 GB RTX 5060 Laptop GPU; concurrent SmolVLA/SAM3 loading used about 5.6 GB. Memory usage depends on configuration. RPent uses Python 3.11; keep the bridge in a separate Python 3.10 or 3.11 environment because the model stacks require different Transformers versions.

Run from the **RPent repository root**. Existing compatible environments can be reused. New installations also need the system packages and SAM3 assets described in the [LIBERO guide](../../../docs/source-en/rst_source/simulators/libero.rst).

```bash
uv venv --python 3.11 .venv-rpent-libero-pro
uv pip install --python .venv-rpent-libero-pro/bin/python -e '.[libero-pro]' --torch-backend cu128
uv venv --python 3.10 .venv-smolvla
uv pip install --python .venv-smolvla/bin/python \
  -r robots/libero/smolvla_bridge/requirements.txt --torch-backend cu128
```

Do not install the bridge requirements into the RPent environment. The bridge pins [LeRobot 0.3.3](https://pypi.org/project/lerobot/0.3.3/); its published SmolVLA core files match the development copy. RPent's `flywheel` extra alone does not supply the bridge's model dependencies.


Prepare the simulator assets and SAM3 separately. After installation, run from the repository root:

```bash
.venv-rpent-libero-pro/bin/liberopro-download-assets --skip-existing
uv pip install --python .venv-rpent-libero-pro/bin/python modelscope
.venv-rpent-libero-pro/bin/modelscope download facebook/sam3 \
  --local-dir checkpoints/sam3
```

Downloads require network access and any access permissions imposed by the upstream models. Existing assets can be referenced in the local configuration without downloading them again.

## Prepare assets

The expected asset layout is:

```text
checkpoints/
  sam3/sam3.pt
  smolvla/
    checkpoint_015000.pt
    base/config.json
    base/model.safetensors
    vlm/config.json
    vlm/tokenizer.json
    vlm/tokenizer_config.json
    vlm/preprocessor_config.json
    vlm/processor_config.json
    vlm/special_tokens_map.json
    vlm/chat_template.json
```

Download the pinned [SmolVLA base](https://huggingface.co/lerobot/smolvla_base) and [SmolVLM processor](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) assets:

```bash
.venv-smolvla/bin/hf download lerobot/smolvla_base \
  --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 \
  --include config.json model.safetensors \
  --local-dir checkpoints/smolvla/base
.venv-smolvla/bin/hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --revision 4cde9d208e19519c36917aa170fffc7e381aa0ad \
  --include '*.json' --local-dir checkpoints/smolvla/vlm
cp robots/libero/smolvla_bridge/processor_overrides/*.json checkpoints/smolvla/vlm/
```

The two small `processor_overrides` JSON files preserve the chat template used during training. Only their chat-template fields differ from the pinned upstream processor configuration. They derive from the Apache-2.0 SmolVLM processor and contain no model weights. Inference uses local assets without downloading more files.

**The task checkpoint has not been published and has no public download URL yet.** The publisher must separately supply `checkpoint_015000.pt` via Hugging Face or a GitHub Release, then fill in `checkpoint_download_url` in `assets_manifest.json`. Until then, a fresh checkout can run the mock check but needs this file from the publisher for real inference. The checkpoint is a PyTorch dictionary containing `trainable`, `stats`, `state_dim`, and `action_dim`, not an arbitrary pretrained-model directory. Training data are not required for inference. Keep weights out of ordinary Git history; `checkpoints/` is ignored.

Prepare SAM3 according to the LIBERO guide. Verify the recorded asset sizes/hashes, then configure your paths:

```bash
python3 robots/libero/smolvla_bridge/check_assets.py --root checkpoints/smolvla
cp robots/libero/smolvla_bridge/env.example.sh robots/libero/smolvla_bridge/env.local.sh
# Edit env.local.sh for your paths, then source it from the repository root:
source robots/libero/smolvla_bridge/env.local.sh
```

The manifest describes the development assets. A different compatible checkpoint needs its own recorded identity; do not call a hash mismatch the same weights. `env.local.sh` is ignored by Git. Keep credentials out of tracked files.

## Check and start the service

A CPU-only interface/shape check needs no weights:

```bash
SMOLVLA_PYTHON="$PWD/.venv-rpent-libero-pro/bin/python" \
  bash robots/libero/smolvla_bridge/run_server.sh \
  --profile "$PWD/robots/libero/smolvla_bridge/profiles/mock_osc7.json" \
  --mock --self-test
```

Expected output shape: `(1, 5, 7)`. Mock actions are placeholders and must not be used for evaluation. Real inference on a synthetic observation can be checked with `bash robots/libero/smolvla_bridge/run_server.sh --self-test`; this does not establish simulator success.

In terminal one, start the real service and wait for the readiness message:

```bash
bash robots/libero/smolvla_bridge/run_server.sh --port 8320
```

Check liveness with `curl http://127.0.0.1:8320/healthz`. Leave this terminal running; stop it with Ctrl+C when finished.

## Run seed 0 with memory

In terminal two, from the repository root:

```bash
source robots/libero/smolvla_bridge/env.local.sh
export OPENAI_BASE_URL='https://your-provider.example/v1'
export RPENT_API_MODEL='openai-chat:your-vision-model'
read -rsp 'API Key: ' OPENAI_API_KEY
echo
export OPENAI_API_KEY
bash robots/libero/smolvla_bridge/run_episode.sh
```

Choose a provider/model that supports images, tool calling, and the requested reasoning effort. RPent starts LIBERO and SAM3 and connects to the existing SmolVLA service. `--am-backend pi05` reuses the client's observation encoding and RPC protocol; the endpoint serves SmolVLA.

The default `memory/` is a small curated subset: the actual successful seed-0 audit/recipe and general localization/grasp lessons. It differs from the full development memory corpus. Set `RPENT_MEMORY_DIR` to use another corpus. Re-localize targets rather than copying historical coordinates.

The three episode improvements are off by default. Enable any desired combination explicitly:

```bash
bash robots/libero/smolvla_bridge/run_episode.sh \
  --episode-brief --compact-observation-images --assess-movement
```

`RPENT_PYTHON` and `SMOLVLA_PYTHON` select existing interpreters; `RPENT_AM_ENDPOINT` changes the bridge address; `RPENT_OUTPUT_DIR` changes the output directory. Extra RPent arguments pass through, such as `--seed 1` or `--reasoning-effort medium`. Start with the default seed 0 for this example.

New `logs/smolvla_seed0_<timestamp>/` directories contain the transcript, observations, trajectory, and video. Require `environment_success=true` / native `terminated=true` for success. A clean exit or positive grasp heuristic is insufficient. RPent cleans up its own env/SAM3 services; stop the independent bridge separately.

## Input contract and validation limits

The profile retains training settings: 8D EEF/gripper state, 7D OSC delta actions, two 256×256 images, chunk size 16, five returned actions, and ten flow steps. The bridge changes image layout/range without rotating images.

The model instruction is **intentionally fixed to `pick up the salad dressing`**. Other planner tool prompts are replaced before model inference. The small checkpoint depends on the training instruction; the API planner retains responsibility for the full placement task.

Validation covers the mock CLI, actual local HTTP calls from RPent's client, image orientation, fixed-prompt replacement, and real-checkpoint inference using the existing environment and published LeRobot 0.3.3 source. A complete installation and episode on a fresh machine remain unverified. A fixed simulator seed does not fix API choices or VLA sampling noise.

[Detailed validation record](VALIDATION.md)
