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

"""SmolVLA 模型加载与推理。

构建方式与训练时一致，读取基础模型、任务权重和归一化统计：

1. 读基础模型目录的 ``config.json``，用本地 VLM 目录替换
   ``vlm_model_name``，并关掉 ``load_vlm_weights``（权重从 safetensors 来）；
2. 用官方 ``model.safetensors`` 填满整个模型；
3. 再用任务 checkpoint 里的 ``trainable`` 覆盖微调后的动作专家；
4. 用 checkpoint 里的 ``stats`` 建归一化层（MEAN_STD）。

这里只读取上述资产，不写回、不复制、不修改任何仓库文件。
"""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .adapters import ObsError, build_batch, postprocess_actions
from .profiles import Profile

Log = Callable[[str], None]


def _noop(_: str) -> None:
    return


class SmolVLABackend:
    """加载一个 SmolVLA checkpoint，并把 RPent 的 obs 变成动作块。"""

    def __init__(
        self,
        profile: Profile,
        *,
        overrides: dict[str, Any] | None = None,
        allow_shape_mismatch: bool = False,
        log: Log = _noop,
    ) -> None:
        self.profile = self._apply_overrides(profile, overrides or {})
        self.allow_shape_mismatch = allow_shape_mismatch
        self.log = log
        self.policy: Any = None
        self._torch: Any = None
        self._dtype: Any = None
        self.predict_count = 0

    # -- 配置 -----------------------------------------------------------------
    @staticmethod
    def _apply_overrides(profile: Profile, overrides: dict[str, Any]) -> Profile:
        allowed = {
            key: value
            for key, value in overrides.items()
            if value is not None
            and key in {f.name for f in dataclasses.fields(Profile)}
        }
        return dataclasses.replace(profile, **allowed) if allowed else profile

    def _inject_paths(self) -> None:
        """Use offline assets and optionally add a legacy LeRobot source tree."""
        env = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        for key, value in env.items():
            os.environ.setdefault(key, value)

        import sys

        for site in reversed(self.profile.assets.python_site_packages):
            if site and site not in sys.path:
                sys.path.append(site)
        src = self.profile.assets.lerobot_src
        if src and src not in sys.path:
            sys.path.insert(0, src)

    # -- 加载 -----------------------------------------------------------------
    def load(self) -> None:
        started = time.time()
        self._inject_paths()

        import torch
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from safetensors.torch import load_file

        self._torch = torch
        self._dtype = getattr(torch, self.profile.dtype)

        profile = self.profile
        assets = profile.assets
        ckpt_path = Path(profile.checkpoint) if profile.checkpoint else None
        base_dir = Path(assets.base_model_dir)

        ckpt: dict[str, Any] | None = None
        if ckpt_path is not None:
            self.log(f"[backend] checkpoint = {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            ckpt_state_dim = ckpt.get("state_dim")
            ckpt_action_dim = ckpt.get("action_dim")
            if ckpt_state_dim and int(ckpt_state_dim) != profile.state_dim:
                raise RuntimeError(
                    f"profile 声明 state_dim={profile.state_dim}，但 checkpoint 是 {ckpt_state_dim}"
                )
            if ckpt_action_dim and int(ckpt_action_dim) != profile.action_dim:
                raise RuntimeError(
                    f"profile 声明 action_dim={profile.action_dim}，但 checkpoint 是 {ckpt_action_dim}"
                )
        else:
            self.log(
                "[backend] 未指定 checkpoint：只加载官方初始权重（等于未微调的 smolvla_base）"
            )

        stats = self._resolve_stats(ckpt)

        raw = dict(__import__("json").loads((base_dir / "config.json").read_text()))
        raw.pop("type", None)
        raw["input_features"] = {
            "observation.state": PolicyFeature(FeatureType.STATE, (profile.state_dim,)),
            **{
                key: PolicyFeature(
                    FeatureType.VISUAL, (3, profile.image_size, profile.image_size)
                )
                for key in profile.camera_keys()
            },
        }
        raw["output_features"] = {
            "action": PolicyFeature(FeatureType.ACTION, (profile.action_dim,))
        }
        raw["normalization_mapping"] = {
            key: NormalizationMode(value)
            for key, value in raw["normalization_mapping"].items()
        }
        resize = int(profile.resize or profile.image_size)
        raw.update(
            vlm_model_name=assets.vlm_dir,
            load_vlm_weights=False,
            resize_imgs_with_padding=(resize, resize),
            chunk_size=profile.chunk_size,
            n_action_steps=profile.n_action_steps,
            tokenizer_max_length=profile.tokenizer_max_length,
            num_steps=profile.num_steps,
            device=profile.device,
            use_amp=False,
        )

        self.log(
            f"[backend] 构建 SmolVLA: state={profile.state_dim} action={profile.action_dim} "
            f"cameras={profile.cameras} chunk={profile.chunk_size}/{profile.n_action_steps} "
            f"steps={profile.num_steps}"
        )
        policy = SmolVLAPolicy(SmolVLAConfig(**raw), dataset_stats=stats)

        # 1) 官方预训练权重（含 VLM 主干与动作专家）
        base_file = base_dir / "model.safetensors"
        if base_file.is_file():
            base_state = load_file(str(base_file))
            base_state = {
                key.replace("model._orig_mod.", "model."): value
                for key, value in base_state.items()
            }
            own = policy.state_dict()
            base_state = {
                key: value
                for key, value in base_state.items()
                if key.startswith("model.") and key in own
            }
            missing, unexpected = policy.load_state_dict(base_state, strict=False)
            self.log(
                f"[backend] 官方权重 {len(base_state)} 项 (missing={len(missing)}, unexpected={len(unexpected)})"
            )
        else:
            raise FileNotFoundError(f"缺少官方权重: {base_file}")

        # 2) 任务 checkpoint 的微调权重
        trainable = (ckpt or {}).get("trainable") or {}
        if not trainable:
            self.log(
                "[backend] 没有微调权重：当前就是官方初始 SmolVLA（未注入任何任务经验）"
            )
        if trainable:
            trainable = {
                key.replace("model._orig_mod.", "model."): value
                for key, value in trainable.items()
            }
            own = policy.state_dict()
            trainable = {key: value for key, value in trainable.items() if key in own}
            missing, unexpected = policy.load_state_dict(trainable, strict=False)
            self.log(
                f"[backend] 微调权重 {len(trainable)} 项 (missing={len(missing)}, unexpected={len(unexpected)})"
            )

        # SmolVLA creates float32 action noise internally. Keep parameters and
        # inputs in float32, matching the trainer, and use autocast for CUDA
        # matrix operations instead of permanently converting the whole model.
        policy.to(device=profile.device)
        policy.eval()
        self.policy = policy
        self.log(f"[backend] 加载完成 {time.time() - started:.1f}s on {profile.device}")

    # -- 归一化统计 -----------------------------------------------------------
    def _resolve_stats(self, ckpt: dict[str, Any] | None) -> dict[str, Any]:
        """决定 MEAN_STD 用的统计量。

        - ``checkpoint``：从任务 checkpoint 的 ``stats`` 取（训练时统计的真实数据分布）
        - ``identity``：均值 0 / 方差 1，等价于“不归一化”；只用于加载官方初始权重做冒烟测试
        - 其它：视为文件路径（.pt 里取 ``stats`` 或直接是 dict；也可以是 .json）
        """
        import json

        import torch

        source = self.profile.stats
        if source == "checkpoint":
            stats = (ckpt or {}).get("stats")
            if not isinstance(stats, dict) or "action" not in stats:
                raise RuntimeError(
                    "checkpoint 里没有可用的归一化 stats（需要 stats['observation.state'] 与 stats['action']）"
                )
            return stats

        if source == "identity":
            self.log(
                "[backend] 警告：stats=identity，状态/动作不做归一化；仅适合验证初始权重能否跑通"
            )
            state_dim, action_dim = self.profile.state_dim, self.profile.action_dim
            eye = {
                "mean": torch.zeros(state_dim),
                "std": torch.ones(state_dim),
            }
            eye_action = {
                "mean": torch.zeros(action_dim),
                "std": torch.ones(action_dim),
            }
            return {"observation.state": eye, "action": eye_action}

        path = Path(source)
        payload: Any
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
        stats = payload.get("stats", payload) if isinstance(payload, dict) else None
        if not isinstance(stats, dict) or "action" not in stats:
            raise RuntimeError(
                f"{path} 里找不到可用的 stats（需要 observation.state 与 action）"
            )
        self.log(f"[backend] 从 {path} 读取归一化统计")
        return stats

    # -- 推理 -----------------------------------------------------------------
    def predict(
        self, obs: dict[str, Any], options: dict[str, Any] | None = None
    ) -> np.ndarray:
        if self.policy is None:
            raise RuntimeError("backend 还没 load()")
        torch = self._torch
        started = time.time()

        batch, state_peak = build_batch(self.profile, obs)
        tensors: dict[str, Any] = {}
        for key, value in batch.items():
            if key == "task":
                tensors[key] = value
            else:
                tensors[key] = torch.from_numpy(np.asarray(value)).to(
                    device=self.profile.device, dtype=torch.float32
                )

        device_type = torch.device(self.profile.device).type
        autocast_enabled = device_type == "cuda" and self._dtype != torch.float32
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device_type,
                dtype=self._dtype,
                enabled=autocast_enabled,
            ),
        ):
            chunk = self.policy.predict_action_chunk(tensors)
        chunk = chunk.float().cpu().numpy()

        actions = postprocess_actions(
            self.profile, chunk, allow_mismatch=self.allow_shape_mismatch
        )
        self.predict_count += 1
        if self.predict_count == 1 or self.predict_count % 20 == 0:
            self.log(
                f"[predict#{self.predict_count}] raw={tuple(chunk.shape)} "
                f"sent={tuple(actions.shape)} |state|max={state_peak:.3f} "
                f"{time.time() - started:.2f}s"
            )
        return actions

    # -- 元信息 ---------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "chunk_size": self.profile.chunk_size,
            "n_action_steps": self.profile.n_action_steps,
            "exec_steps": self.profile.exec_steps,
            "state_dim": self.profile.state_dim,
            "action_dim": self.profile.action_dim,
            "predictions": self.predict_count,
        }

    def model_meta(self) -> dict[str, Any]:
        return {
            "backend": "smolvla",
            "profile": self.profile.name,
            "checkpoint": self.profile.checkpoint,
            "task_prompt": self.profile.task_prompt,
            "state_schema": self.profile.state_schema,
            "action_schema": self.profile.action_schema,
            "env_action_dim": self.profile.env_action_dim,
            "cameras": self.profile.cameras,
            "notes": self.profile.notes,
        }

    def close(self) -> None:
        self.policy = None


class MockBackend:
    """不加载模型，只按 profile 返回占位动作块 —— 用于验证协议与维度链路。"""

    def __init__(self, profile: Profile, *, log: Log = _noop) -> None:
        self.profile = profile
        self.log = log
        self.predict_count = 0

    def load(self) -> None:
        self.log(
            f"[mock] 未加载真实模型；将按 profile 返回 {self.profile.exec_steps}x"
            f"{self.profile.env_action_dim or self.profile.action_dim} 的占位动作"
        )

    def predict(
        self, obs: dict[str, Any], options: dict[str, Any] | None = None
    ) -> np.ndarray:
        build_batch(self.profile, obs)  # 仍然校验观测，便于提前发现字段问题
        self.predict_count += 1
        dim = self.profile.env_action_dim or self.profile.action_dim
        ramp = np.linspace(0.0, 0.01, self.profile.exec_steps, dtype=np.float32)
        return np.repeat(ramp[:, None], dim, axis=1)[None]

    def status(self) -> dict[str, Any]:
        return {"mode": "mock", "exec_steps": self.profile.exec_steps}

    def model_meta(self) -> dict[str, Any]:
        return {"backend": "mock", "profile": self.profile.name}

    def close(self) -> None:
        return


def build_backend(
    profile: Profile,
    *,
    mock: bool = False,
    overrides: dict[str, Any] | None = None,
    allow_shape_mismatch: bool = False,
    log: Log = _noop,
) -> Any:
    if mock:
        return MockBackend(
            SmolVLABackend._apply_overrides(profile, overrides or {}), log=log
        )
    return SmolVLABackend(
        profile,
        overrides=overrides,
        allow_shape_mismatch=allow_shape_mismatch,
        log=log,
    )


__all__ = ["SmolVLABackend", "MockBackend", "build_backend", "ObsError"]
