# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Portable configuration and the standalone CLI interoperate with RPent RPC."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from robots.libero.smolvla_bridge.bridge.profiles import Profile
from rpent.robots.components.pi05_vla_client import Pi05VLAClient
from rpent.utils.rpc.http_rpc import HttpRpcClient

BRIDGE = Path(__file__).resolve().parents[4] / "robots/libero/smolvla_bridge"


def test_mock_cli_runs_without_weights_from_another_working_directory(tmp_path):
    env = os.environ.copy()
    for key in ["SMOLVLA_OSC7_CHECKPOINT", "SMOLVLA_BASE_DIR", "SMOLVLA_VLM_DIR"]:
        env.pop(key, None)
    result = subprocess.run(
        [
            sys.executable,
            str(BRIDGE / "server.py"),
            "--profile",
            str(BRIDGE / "profiles/mock_osc7.json"),
            "--mock",
            "--self-test",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "(1, 5, 7)" in result.stdout


def test_missing_environment_variable_reports_the_variable(monkeypatch):
    monkeypatch.delenv("SMOLVLA_OSC7_CHECKPOINT", raising=False)
    with pytest.raises(ValueError, match="SMOLVLA_OSC7_CHECKPOINT"):
        Profile.load(BRIDGE / "profiles/libero_pro_salad.json")


def test_asset_paths_are_portable_and_missing_files_fail_early(tmp_path, monkeypatch):
    base, vlm = tmp_path / "base", tmp_path / "vlm"
    base.mkdir()
    vlm.mkdir()
    for name in ["config.json", "model.safetensors"]:
        (base / name).write_text("test")
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
    ]:
        (vlm / name).write_text("test")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("test")
    monkeypatch.setenv("SMOLVLA_OSC7_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("SMOLVLA_BASE_DIR", str(base))
    monkeypatch.setenv("SMOLVLA_VLM_DIR", str(vlm))
    p = Profile.load(BRIDGE / "profiles/libero_pro_salad.json")
    assert p.assets.lerobot_src == ""
    assert p.assets.base_model_dir == str(base)
    assert p.task_prompt == "pick up the salad dressing"
    (vlm / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        Profile.load(BRIDGE / "profiles/libero_pro_salad.json")


def test_real_rpent_client_calls_standalone_mock_and_shuts_it_down(tmp_path):
    log = tmp_path / "server.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                str(BRIDGE / "server.py"),
                "--profile",
                str(BRIDGE / "profiles/mock_osc7.json"),
                "--mock",
                "--port",
                "0",
            ],
            cwd=tmp_path,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 15
            url = None
            while time.monotonic() < deadline:
                text = log.read_text()
                for line in text.splitlines():
                    if "服务就绪: " in line:
                        url = line.split("服务就绪: ", 1)[1].removesuffix("/call")
                if url:
                    break
                assert process.poll() is None, text
                time.sleep(0.05)
            assert url, log.read_text()
            rpc = HttpRpcClient(url)
            assert rpc.call("healthz")["status"] == "ok"
            assert rpc.call("vla.get_model_meta")["backend"] == "mock"
            model = Pi05VLAClient(rpc, embodiment="libero")
            actions = model.predict(
                {
                    "main_images": np.zeros((256, 256, 3), dtype=np.uint8),
                    "wrist_images": np.zeros((256, 256, 3), dtype=np.uint8),
                    "states": np.zeros(8, dtype=np.float32),
                    "task_descriptions": "arbitrary planner wording",
                }
            )
            assert actions.shape == (5, 7)
            assert actions.dtype == np.float32
            assert np.isfinite(actions).all()
            assert rpc.call("shutdown") == {"ok": True}
            assert process.wait(timeout=10) == 0
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


def test_published_processor_overrides_match_the_training_asset_manifest():
    manifest = json.loads((BRIDGE / "assets_manifest.json").read_text())
    for name in manifest["processor_overrides"]:
        data = (BRIDGE / "processor_overrides" / name).read_bytes()
        assert (
            hashlib.sha256(data).hexdigest()
            == manifest["files"]["vlm/" + name]["sha256"]
        )
