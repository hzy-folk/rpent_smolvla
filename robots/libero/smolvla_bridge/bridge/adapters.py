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

"""观测/动作适配：RPent 的 Pi0.5 VLA 线格式 <-> SmolVLA 的 batch 格式。

RPent 侧（``pi05_vla_client._encode_obs_libero``）发来的字段是固定的：

    main_images        [B, H, W, 3] uint8  主视角(agentview)
    wrist_images       [B, H, W, 3] uint8 或 None
    extra_view_images  [B, H, W, 3] uint8 或 None
    states             [B, D] float32     RPent 自己的 state 向量
    task_descriptions  [str]

SmolVLA 侧（lerobot ``SmolVLAPolicy.predict_action_chunk``）需要：

    observation.state            [B, state_dim] float32，未归一化
    observation.images.cameraN   [B, 3, H, W] float32，取值范围 [0, 1]
    task                         list[str]

两者之间没有任何“语义”约定 —— 维度、含义都由 profile 声明。这里只做形状搬运
与显式报错，绝不猜测。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .profiles import Profile


class ObsError(ValueError):
    """观测无法按当前 profile 适配。"""


class ActionError(ValueError):
    """模型输出无法按当前 profile 交给环境。"""


# ---------------------------------------------------------------------------
# 观测
# ---------------------------------------------------------------------------


def _to_float_chw(images: Any, *, name: str, image_size: int) -> np.ndarray:
    """[B,H,W,3] uint8/float -> [B,3,H,W] float32 in [0,1]。"""
    if images is None:
        raise ObsError(f"{name} 缺失")
    arr = np.asarray(images)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ObsError(f"{name} 需要 [B,H,W,3]，收到 {arr.shape}")
    if arr.shape[1:3] != (image_size, image_size):
        raise ObsError(f"{name} 需要 [B,{image_size},{image_size},3]，收到 {arr.shape}")
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        peak = float(np.nanmax(arr)) if arr.size else 0.0
        if peak > 1.5:  # 传进来的是 0-255 的 float
            arr = arr / 255.0
    if not np.isfinite(arr).all():
        raise ObsError(f"{name} 含 NaN/Inf")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ObsError(
            f"{name} 转换后像素值必须在 [0,1]，收到范围 [{arr.min()}, {arr.max()}]"
        )
    return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))


def _fit_state(states: Any, profile: Profile) -> np.ndarray:
    """按 profile 的 state_adapter 把状态对齐到 state_dim。"""
    if states is None:
        raise ObsError("states 缺失")
    arr = np.asarray(states, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.ndim != 2:
        raise ObsError(f"states 需要 [B,D]，收到 {arr.shape}")
    if not np.isfinite(arr).all():
        raise ObsError("states 含 NaN/Inf")

    got, want = arr.shape[1], profile.state_dim
    if got == want:
        return arr
    mode = profile.state_adapter
    if mode == "strict":
        raise ObsError(
            f"状态维度不匹配：RPent 发来 {got} 维，但 profile {profile.name!r} 需要 {want} 维。\n"
            f"  该 checkpoint 的 state 定义: {profile.state_schema or '(未填写)'}\n"
            f"  如果只是验证链路，可把 profile 的 state_adapter 改成 'pad' 或 'truncate'，"
            f"但那样得到的动作没有物理意义。"
        )
    if mode == "truncate":
        return arr[:, :want]
    if got < want:
        pad = np.zeros((arr.shape[0], want - got), dtype=np.float32)
        return np.concatenate([arr, pad], axis=1)
    return arr[:, :want]


def build_batch(profile: Profile, obs: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """把 RPent 的 obs 字典转成 SmolVLA batch（numpy，尚未转 tensor）。

    返回 ``(batch, max_state_abs)``，后者用于日志里的健康检查。
    """
    if not isinstance(obs, dict):
        raise ObsError(f"obs 必须是 dict，收到 {type(obs).__name__}")
    if "task_descriptions" not in obs:
        raise ObsError("obs 缺少 task_descriptions")

    batch: dict[str, Any] = {}
    batch["observation.images.camera1"] = _to_float_chw(
        obs.get("main_images"), name="main_images", image_size=profile.image_size
    )
    if profile.cameras >= 2:
        batch["observation.images.camera2"] = _to_float_chw(
            obs.get("wrist_images"), name="wrist_images", image_size=profile.image_size
        )
    if profile.cameras >= 3:
        extra = obs.get("extra_view_images")
        if extra is not None:
            batch["observation.images.camera3"] = _to_float_chw(
                extra, name="extra_view_images", image_size=profile.image_size
            )
        else:  # 允许用第三视角缺省时复用主视角，避免直接失败
            batch["observation.images.camera3"] = batch["observation.images.camera1"]

    state = _fit_state(obs.get("states"), profile)
    batch["observation.state"] = state

    incoming_task = obs["task_descriptions"]
    if isinstance(incoming_task, str):
        incoming_task = [incoming_task]
    incoming_task = [str(item) for item in incoming_task]
    task = (
        [profile.task_prompt] * len(incoming_task)
        if profile.task_prompt
        else incoming_task
    )
    if isinstance(task, str):
        task = [task]
    batch["task"] = [str(item) for item in task]

    # 批维必须一致
    sizes = {
        key: value.shape[0]
        for key, value in batch.items()
        if key.startswith("observation.")
    }
    if len(set(sizes.values())) != 1:
        raise ObsError(f"批维不一致: {sizes}")
    return batch, float(np.max(np.abs(state)))


# ---------------------------------------------------------------------------
# 动作
# ---------------------------------------------------------------------------


def postprocess_actions(
    profile: Profile, chunk: np.ndarray, *, allow_mismatch: bool = False
) -> np.ndarray:
    """返回 Pi05 RPC 约定的单环境动作块 ``[1, steps, action_dim]``。"""
    actions = np.asarray(chunk, dtype=np.float32)
    if actions.ndim == 3:  # [B, T, A] -> 取第 0 个 env
        if actions.shape[0] != 1:
            raise ActionError(
                f"LIBERO bridge 只接受单环境 batch，收到 batch={actions.shape[0]}"
            )
        actions = actions[0]
    if actions.ndim != 2:
        raise ActionError(f"动作块需要 [T, A]，收到 {actions.shape}")
    if actions.shape[0] < profile.exec_steps:
        raise ActionError(
            f"模型只预测了 {actions.shape[0]} 步，profile 要求至少 {profile.exec_steps} 步"
        )
    if not np.isfinite(actions).all():
        raise ActionError("动作含 NaN/Inf")

    target_dim = profile.env_action_dim
    if target_dim is not None and actions.shape[1] != target_dim:
        if profile.action_adapter == "strict" and not allow_mismatch:
            raise ActionError(
                f"动作维度不匹配：模型输出 {actions.shape[1]} 维，"
                f"RPent 环境需要 {target_dim} 维。\n"
                f"  该 checkpoint 的 action 定义: {profile.action_schema or '(未填写)'}\n"
                f"  这不是形状问题而是动作语义问题（关节绝对位置 vs 末端 OSC 增量），"
                f"用 pad/truncate 只能验证链路，不能真正控制机械臂。"
            )
        if actions.shape[1] > target_dim:
            actions = actions[:, :target_dim]
        else:
            pad = np.zeros(
                (actions.shape[0], target_dim - actions.shape[1]), np.float32
            )
            actions = np.concatenate([actions, pad], axis=1)

    if profile.action_bounds is not None:
        low, high = profile.action_bounds
        actions = np.clip(actions, low, high)

    # Pi05VLAClient.predict removes the leading environment batch with [0].
    # Keep it on the RPC wire so that the client receives [steps, action_dim].
    return np.ascontiguousarray(actions[: profile.exec_steps][None], dtype=np.float32)
