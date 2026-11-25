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

"""Observation request projection keeps evidence and leaves source messages intact."""

from copy import deepcopy

from pydantic_ai import BinaryContent, ModelRequest, UserPromptPart

from rpent.planner.episode_context import ObservationImages


def frames(messages):
    return [
        item.data
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
        for item in part.content
        if isinstance(item, BinaryContent)
    ]


def add(policy, messages, step, action="move_to", name=None, **result):
    images = policy.attach(
        name or action,
        {
            "step": step,
            "state": {},
            "log": {
                "command": {"action": action},
                "result": {"final_dist_m": 0.001, **result},
            },
        },
        [BinaryContent(f"step-{step}".encode(), media_type="image/png")],
    )
    messages.append(ModelRequest(parts=[UserPromptPart(images)]))


def test_old_normal_images_drop_but_key_pairs_and_originals_remain():
    policy, messages = ObservationImages(), []
    add(policy, messages, 0, name="view_env_state")
    add(policy, messages, 1)
    add(policy, messages, 2, "pi0_pick")
    add(policy, messages, 3)
    add(policy, messages, 4)
    source = deepcopy(messages)
    projected = policy.project(messages)
    assert frames(projected) == [b"step-0", b"step-1", b"step-2", b"step-4"]
    assert frames(messages) == frames(source)


def test_explicit_historical_read_survives_next_request_without_rewinding_latest():
    policy, messages = ObservationImages(), []
    add(policy, messages, 0)
    add(policy, messages, 1)
    policy.project(messages)
    add(policy, messages, 0, name="view_env_state")
    result = policy.project(messages)
    assert b"step-1" in frames(result)
    assert frames(result).count(b"step-0") == 2


def test_incomplete_move_keeps_before_after_and_event_budget_is_bounded():
    policy, messages = ObservationImages(), []
    for step in range(10):
        add(policy, messages, step, "pi0_pick" if step % 2 else "move_to")
    add(policy, messages, 10, final_dist_m=0.1)
    add(policy, messages, 11)
    result = frames(policy.project(messages))
    assert b"step-9" in result and b"step-10" in result
    assert b"step-11" in result
    assert b"step-1" not in result
    assert len(result) <= 6


def test_reset_discards_previous_episode_pins():
    policy, messages = ObservationImages(), []
    add(policy, messages, 0)
    add(policy, messages, 1, "pi0_pick")
    add(policy, messages, 2, "reset")
    assert frames(policy.project(messages)) == [b"step-2"]


def test_segmentation_overlay_is_delivered_without_becoming_latest_observation():
    policy, messages = ObservationImages(), []
    add(policy, messages, 2)
    overlay = policy.attach(
        "segment",
        {"step": 2, "score": 0.9},
        [BinaryContent(b"overlay", media_type="image/png")],
    )
    messages.append(ModelRequest(parts=[UserPromptPart(overlay)]))
    assert frames(policy.project(messages)) == [b"step-2", b"overlay"]
    assert frames(policy.project(messages)) == [b"step-2"]
