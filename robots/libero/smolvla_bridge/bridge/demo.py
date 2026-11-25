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

"""构造与 RPent Pi0.5 客户端同形状的假观测，用于自检与协议联调。"""

from __future__ import annotations

import numpy as np

from .profiles import Profile


def fake_obs(
    profile: Profile,
    *,
    state_dim: int | None = None,
    batch: int = 1,
    instruction: str = "pick up the black bowl",
    seed: int = 0,
) -> dict:
    """返回 ``main_images/wrist_images/extra_view_images/states/task_descriptions``。"""
    rng = np.random.default_rng(seed)
    h = w = profile.image_size
    img = (rng.random((batch, h, w, 3)) * 255).astype(np.uint8)
    dim = state_dim if state_dim is not None else profile.state_dim
    return {
        "main_images": img,
        "wrist_images": img.copy() if profile.cameras >= 2 else None,
        "extra_view_images": img.copy() if profile.cameras >= 3 else None,
        "states": np.zeros((batch, dim), dtype=np.float32),
        "task_descriptions": [instruction] * batch,
    }
