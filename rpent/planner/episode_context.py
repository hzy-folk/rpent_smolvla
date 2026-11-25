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

"""Opt-in request projection for observation images and a single current brief."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic_ai import BinaryContent, ModelRequest, UserPromptPart
from pydantic_ai.messages import ModelMessage, TextContent, ToolReturnPart

BRIEF_MARKER = "rpent.episode_brief"


def inject_brief(messages: list[ModelMessage], text: str) -> list[ModelMessage]:
    """Replace our previous tagged brief without removing tool call/result pairs."""
    projected = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            projected.append(message)
            continue
        parts = []
        for part in message.parts:
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                content = [
                    item
                    for item in part.content
                    if not (
                        isinstance(item, TextContent) and item.metadata == BRIEF_MARKER
                    )
                ]
                if content:
                    parts.append(replace(part, content=content))
            else:
                parts.append(part)
        if parts:
            projected.append(replace(message, parts=parts))
    part = UserPromptPart([TextContent(text, metadata=BRIEF_MARKER)])
    if projected and isinstance(projected[-1], ModelRequest):
        projected[-1] = replace(projected[-1], parts=[*projected[-1].parts, part])
    else:
        projected.append(ModelRequest(parts=[part]))
    return projected


@dataclass
class _Images:
    step: int | None
    explicit: bool = False
    expires: int = 0


class ObservationImages:
    """Keep latest views and bounded event pairs; leave untagged images alone.

    Only image attachments are projected. Artifacts, tool text and canonical
    capture remain intact. Explicit reads survive the next request.
    """

    def __init__(self, *, move_tol: float = 0.012, yaw_tol: float = 0.02) -> None:
        self._lock = RLock()
        self.move_tol = move_tol
        self.yaw_tol = yaw_tol
        self._images: dict[str, str] = {}
        self._groups: dict[str, _Images] = {}
        self._events: deque[tuple[str, ...]] = deque(maxlen=2)
        self._latest: str | None = None
        self._initial: str | None = None
        self._request = 0

    def attach(
        self,
        name: str,
        data: dict[str, Any],
        images: list[BinaryContent],
    ) -> list[BinaryContent]:
        """Tag one feedback image group; a historical read never replaces latest."""
        with self._lock:
            return self._attach(name, data, images)

    def _attach(self, name, data, images):
        if not images:
            return images
        command = data.get("log", {}).get("command") or {}
        result = data.get("log", {}).get("result") or {}
        # Overlays and other non-observation attachments must be delivered at least once.
        explicit = name in {"read_image", "view_env_state"} or "state" not in data
        if command.get("action") == "reset" and not explicit:
            self._events.clear()
            for old in self._groups.values():
                old.expires = 0
            self._latest = self._initial = None
        group = uuid4().hex
        step = data.get("step")
        self._groups[group] = _Images(step, explicit, self._request + 1)
        previous = self._latest
        latest_step = self._groups[previous].step if previous else None
        is_observation = "state" in data
        if (
            is_observation
            and step is not None
            and (latest_step is None or step >= latest_step)
        ):
            if not explicit or latest_step is None or step > latest_step:
                self._latest = group
        if self._initial is None and is_observation:
            self._initial = group
        critical = bool(data.get("error") or result.get("error"))
        action = command.get("action")
        critical |= action in {"pi0_pick", "pi0_doubled", "release"}
        if action == "move_to":
            assessment = result.get("motion_assessment")
            if assessment:
                critical |= assessment["status"] != "completed"
            else:
                # A rounded boundary is retained conservatively, never declared a failure.
                distance = result.get("final_dist_m")
                critical |= distance is None or distance >= command.get(
                    "tol", self.move_tol
                )
                yaw = result.get("final_yaw_error_rad")
                if yaw is not None:
                    critical |= abs(yaw) >= command.get("yaw_tol", self.yaw_tol)
        if critical and not explicit:
            self._events.append(tuple(item for item in (previous, group) if item))
        tagged = []
        for image in images:
            identifier = "rpent-observation-" + uuid4().hex
            self._images[identifier] = group
            tagged.append(
                BinaryContent(
                    image.data,
                    media_type=image.media_type,
                    identifier=identifier,
                    vendor_metadata=image.vendor_metadata,
                )
            )
        return tagged

    def project(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """Omit older routine images from this request, not their saved artifacts."""
        with self._lock:
            return self._project(messages)

    def _project(self, messages):
        self._request += 1
        keep = {self._latest, self._initial}
        keep.update(group for event in self._events for group in event)
        keep.update(
            key
            for key, group in self._groups.items()
            if group.explicit and group.expires >= self._request
        )
        output = []
        for message in messages:
            if not isinstance(message, ModelRequest):
                output.append(message)
                continue
            parts = []
            for part in message.parts:
                if isinstance(part, (UserPromptPart, ToolReturnPart)) and isinstance(
                    part.content, (list, tuple)
                ):
                    content = []
                    for item in part.content:
                        group = (
                            self._images.get(item.identifier)
                            if isinstance(item, BinaryContent)
                            else None
                        )
                        if group is not None and group not in keep:
                            step = self._groups[group].step
                            content.append(
                                f"[Historical image at step {step} omitted; use read_image or view_env_state to retrieve.]"
                            )
                        else:
                            content.append(item)
                    parts.append(replace(part, content=content))
                else:
                    parts.append(part)
            output.append(replace(message, parts=parts))
        return output
