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

"""运行配置档：一个 checkpoint 一份 JSON，描述维度、相机与解码参数。

配置档只描述“这个 checkpoint 是什么”，不做任何维度变换 —— 变换在 adapters.py，
由 ``state_adapter`` / ``action_adapter`` 控制。默认 ``strict``：维度不符就报错，
而不是悄悄截断（悄悄截断会让机器人做出看起来正常但物理上错误的动作）。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ADAPTER_MODES = ("strict", "pad", "truncate")


def _expand(value: str, base: Path) -> str:
    """展开环境变量；相对路径按配置档所在目录解析。"""
    text = os.path.expandvars(os.path.expanduser(str(value)))
    if re.search(r"\$(?:\w+|\{[^}]+\})", text):
        raise ValueError(f"Unset environment variable in asset path: {value}")
    path = Path(text)
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


@dataclass(frozen=True)
class Assets:
    """模型与代码资产的位置（只读引用，不会写回）。"""

    lerobot_src: str = ""
    python_site_packages: tuple[str, ...] = ()
    base_model_dir: str = ""
    vlm_dir: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base: Path) -> "Assets":
        return cls(
            lerobot_src=_expand(raw["lerobot_src"], base)
            if raw.get("lerobot_src")
            else "",
            python_site_packages=tuple(
                _expand(item, base) for item in raw.get("python_site_packages", [])
            ),
            base_model_dir=_expand(raw["base_model_dir"], base)
            if raw.get("base_model_dir")
            else "",
            vlm_dir=_expand(raw["vlm_dir"], base) if raw.get("vlm_dir") else "",
        )


@dataclass(frozen=True)
class Profile:
    name: str
    #: 任务 checkpoint（含 trainable/stats）；为 None 表示只加载官方初始权重
    checkpoint: str | None
    assets: Assets
    state_dim: int
    action_dim: int
    #: 归一化统计来源："checkpoint" | "identity" | 指向 .pt/.json 的路径
    stats: str = "checkpoint"
    cameras: int = 2
    image_size: int = 256
    #: 模型内部 resize 目标；None 表示与 image_size 相同
    resize: int | None = None
    chunk_size: int = 16
    exec_steps: int = 8
    n_action_steps: int = 8
    num_steps: int = 4
    tokenizer_max_length: int = 64
    state_adapter: str = "strict"
    action_adapter: str = "strict"
    env_action_dim: int | None = None
    action_bounds: tuple[float, float] | None = None
    device: str = "cpu"
    dtype: str = "float32"
    state_schema: str = ""
    action_schema: str = ""
    #: 可选固定模型指令；None 时透传 RPent 的 task_descriptions
    task_prompt: str | None = None
    notes: str = ""
    source: str = field(default="", compare=False)

    # -- 校验 -----------------------------------------------------------------
    def validate(self, *, require_assets: bool = True) -> None:
        for mode_field in ("state_adapter", "action_adapter"):
            mode = getattr(self, mode_field)
            if mode not in ADAPTER_MODES:
                raise ValueError(
                    f"{self.name}: {mode_field}={mode!r} 不支持，可选 {ADAPTER_MODES}"
                )
        if self.cameras not in (1, 2, 3):
            raise ValueError(f"{self.name}: cameras 必须是 1/2/3，收到 {self.cameras}")
        if not 1 <= self.exec_steps <= self.chunk_size:
            raise ValueError(
                f"{self.name}: 需要 1 <= exec_steps({self.exec_steps}) <= chunk_size({self.chunk_size})"
            )
        if self.action_bounds is not None:
            low, high = self.action_bounds
            if not low < high:
                raise ValueError(f"{self.name}: action_bounds must satisfy low < high")
        if not require_assets:
            return
        if self.checkpoint and not Path(self.checkpoint).is_file():
            raise FileNotFoundError(
                f"{self.name}: checkpoint 不存在: {self.checkpoint}"
            )
        if not self.checkpoint and self.stats == "checkpoint":
            raise ValueError(
                f"{self.name}: 没有 checkpoint 时 stats 不能取 'checkpoint'，请设为 'identity' 或指向统计文件"
            )
        if (
            self.stats not in ("checkpoint", "identity")
            and not Path(self.stats).is_file()
        ):
            raise FileNotFoundError(f"{self.name}: 统计文件不存在: {self.stats}")
        if self.assets.lerobot_src and not Path(self.assets.lerobot_src).is_dir():
            raise FileNotFoundError(
                f"{self.name}: lerobot_src 不存在: {self.assets.lerobot_src}"
            )
        if self.assets.base_model_dir and not Path(self.assets.base_model_dir).is_dir():
            raise FileNotFoundError(
                f"{self.name}: base_model_dir 不存在: {self.assets.base_model_dir}"
            )
        for name in ("config.json", "model.safetensors"):
            if not (Path(self.assets.base_model_dir) / name).is_file():
                raise FileNotFoundError(f"Missing SmolVLA base asset: {name}")
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "processor_config.json",
        ):
            if not (Path(self.assets.vlm_dir) / name).is_file():
                raise FileNotFoundError(f"Missing SmolVLM processor asset: {name}")

    def camera_keys(self) -> list[str]:
        return [f"observation.images.camera{i}" for i in range(1, self.cameras + 1)]

    # -- 加载 -----------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, *, require_assets: bool = True) -> "Profile":
        source = Path(path).resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        base = source.parent
        checkpoint = raw.get("checkpoint")
        stats = str(raw.get("stats", "checkpoint"))
        profile = cls(
            name=raw.get("name") or source.stem,
            checkpoint=None if not checkpoint else _expand(checkpoint, base),
            assets=Assets.from_dict(raw["assets"], base),
            state_dim=int(raw["state_dim"]),
            action_dim=int(raw["action_dim"]),
            stats=stats
            if stats in ("checkpoint", "identity")
            else _expand(stats, base),
            cameras=int(raw.get("cameras", 2)),
            image_size=int(raw.get("image_size", 256)),
            resize=None if raw.get("resize") is None else int(raw["resize"]),
            chunk_size=int(raw.get("chunk_size", 16)),
            exec_steps=int(raw.get("exec_steps", 8)),
            n_action_steps=int(raw.get("n_action_steps", 8)),
            num_steps=int(raw.get("num_steps", 4)),
            tokenizer_max_length=int(raw.get("tokenizer_max_length", 64)),
            state_adapter=str(raw.get("state_adapter", "strict")),
            action_adapter=str(raw.get("action_adapter", "strict")),
            env_action_dim=(
                None
                if raw.get("env_action_dim") is None
                else int(raw["env_action_dim"])
            ),
            action_bounds=(
                None
                if raw.get("action_bounds") is None
                else (float(raw["action_bounds"][0]), float(raw["action_bounds"][1]))
            ),
            device=str(raw.get("device", "cpu")),
            dtype=str(raw.get("dtype", "float32")),
            state_schema=str(raw.get("state_schema", "")),
            action_schema=str(raw.get("action_schema", "")),
            task_prompt=(
                None if raw.get("task_prompt") is None else str(raw["task_prompt"])
            ),
            notes=str(raw.get("notes", "")),
            source=str(source),
        )
        profile.validate(require_assets=require_assets)
        return profile
