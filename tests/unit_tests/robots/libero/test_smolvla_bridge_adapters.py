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

import unittest

import numpy as np

from robots.libero.smolvla_bridge.bridge.adapters import (
    ActionError,
    build_batch,
    postprocess_actions,
)
from robots.libero.smolvla_bridge.bridge.profiles import Assets, Profile


def make_profile(**overrides):
    values = {
        "name": "test-osc7",
        "checkpoint": None,
        "assets": Assets(lerobot_src="/tmp"),
        "state_dim": 8,
        "action_dim": 7,
        "stats": "identity",
        "cameras": 2,
        "image_size": 256,
        "chunk_size": 16,
        "exec_steps": 5,
        "env_action_dim": 7,
        "action_bounds": (-1.0, 1.0),
        "task_prompt": "pick up the salad dressing",
    }
    values.update(overrides)
    return Profile(**values)


class Osc7AdapterTests(unittest.TestCase):
    def test_images_change_layout_and_scale_without_rotation(self):
        main = np.zeros((1, 256, 256, 3), dtype=np.uint8)
        wrist = np.zeros_like(main)
        main[0, 0, 0, 0] = 255
        obs = {
            "main_images": main,
            "wrist_images": wrist,
            "states": np.zeros((1, 8), dtype=np.float32),
            "task_descriptions": ["pick up the salad dressing"],
        }

        batch, _ = build_batch(make_profile(), obs)

        self.assertEqual(batch["observation.images.camera1"].shape, (1, 3, 256, 256))
        self.assertEqual(batch["observation.images.camera2"].shape, (1, 3, 256, 256))
        self.assertEqual(float(batch["observation.images.camera1"][0, 0, 0, 0]), 1.0)
        self.assertEqual(float(batch["observation.images.camera1"][0, 0, -1, -1]), 0.0)
        self.assertEqual(batch["observation.state"].shape, (1, 8))
        self.assertEqual(batch["task"], ["pick up the salad dressing"])

    def test_replaces_any_incoming_prompt_with_checkpoint_prompt(self):
        obs = {
            "main_images": np.zeros((1, 256, 256, 3), dtype=np.uint8),
            "wrist_images": np.zeros((1, 256, 256, 3), dtype=np.uint8),
            "states": np.zeros((1, 8), dtype=np.float32),
            "task_descriptions": ["Pick the salad dressing and place it in the basket"],
        }
        batch, _ = build_batch(make_profile(), obs)
        self.assertEqual(batch["task"], ["pick up the salad dressing"])

    def test_replaces_other_task_text_too_when_profile_is_fixed(self):
        obs = {
            "main_images": np.zeros((1, 256, 256, 3), dtype=np.uint8),
            "wrist_images": np.zeros((1, 256, 256, 3), dtype=np.uint8),
            "states": np.zeros((1, 8), dtype=np.float32),
            "task_descriptions": ["pick up the BBQ sauce"],
        }
        batch, _ = build_batch(make_profile(), obs)
        self.assertEqual(batch["task"], ["pick up the salad dressing"])

    def test_returns_five_bounded_actions_with_rpc_batch(self):
        prediction = np.linspace(-1.5, 1.5, 16 * 7, dtype=np.float32).reshape(1, 16, 7)

        actions = postprocess_actions(make_profile(), prediction)

        self.assertEqual(actions.shape, (1, 5, 7))
        self.assertTrue(np.all(actions >= -1.0))
        self.assertTrue(np.all(actions <= 1.0))
        np.testing.assert_array_equal(actions[0], np.clip(prediction[0, :5], -1.0, 1.0))

    def test_rejects_short_action_chunk(self):
        with self.assertRaises(ActionError):
            postprocess_actions(make_profile(), np.zeros((1, 4, 7), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
