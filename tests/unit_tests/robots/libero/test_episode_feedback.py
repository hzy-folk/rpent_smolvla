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

"""Short briefs report measurements rather than invented grasp conclusions."""

from types import SimpleNamespace

from robots.libero.episode_feedback import render_brief
from rpent.session.base import StepRecord


def test_false_grasp_heuristic_is_not_declared_failed_or_held():
    record = StepRecord(
        step_idx=8,
        state={
            "robot0_eef_pos": [0.1419, 0.0528, 0.242],
            "robot0_gripper_qpos": [0.0183, -0.0183],
        },
        command={"action": "pi0_pick"},
        result={"success": False, "diagnostics": {"descent_done": False}},
    )
    brief = render_brief(SimpleNamespace(records=lambda: [record]))
    assert len(brief.splitlines()) == 4
    assert len(brief) < 500
    assert "pick_heuristic=false" in brief
    assert "holding unverified" in brief
    assert "0.0366m" in brief
    assert "grasp failed" not in brief


def test_native_success_overrides_pending_grasp_caution():
    record = StepRecord(
        step_idx=13, state={}, terminated=True, command={"action": "release"}, result={}
    )
    assert "native environment success" in render_brief(
        SimpleNamespace(records=lambda: [record])
    )
