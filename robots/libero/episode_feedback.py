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

"""Optional numeric movement feedback and compact, factual episode briefs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MotionMonitor:
    """Assess moves without changing control; track comparable stalled calls.

    Defaults are starting thresholds, not calibrated success-rate claims.
    Read-only calls leave the sequence alone; other actions reset it.
    """

    min_progress_m: float = 0.002
    min_progress_fraction: float = 0.02
    target_tolerance_m: float = 0.005
    yaw_progress_rad: float = 0.005
    _reference: tuple | None = None
    _streak: int = 0

    def __post_init__(self) -> None:
        for value in (
            self.min_progress_m,
            self.target_tolerance_m,
            self.yaw_progress_rad,
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Movement thresholds must be finite and positive")
        if (
            not np.isfinite(self.min_progress_fraction)
            or not 0 <= self.min_progress_fraction <= 1
        ):
            raise ValueError("Movement progress fraction must be between 0 and 1")

    def reset(self) -> None:
        """Forget the current sequence after another action or an episode reset."""
        self._reference = None
        self._streak = 0

    def assess(
        self,
        *,
        target: np.ndarray,
        start_error: float,
        end_error: float,
        tol: float,
        gripper: float,
        step_clip: float,
        action_scale: float,
        target_yaw: float | None,
        start_yaw_error: float | None,
        end_yaw_error: float | None,
        yaw_tol: float,
        yaw_step_clip: float,
        executed_steps: int,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        """Return a full-precision assessment of one completed handler call."""
        values = [
            *target,
            start_error,
            end_error,
            tol,
            gripper,
            step_clip,
            action_scale,
            yaw_tol,
            yaw_step_clip,
        ]
        if target_yaw is not None:
            values.extend([target_yaw, start_yaw_error, end_yaw_error])
        valid = all(value is not None and np.isfinite(value) for value in values)
        threshold = max(self.min_progress_m, self.min_progress_fraction * start_error)
        position_done = end_error < tol
        yaw_done = target_yaw is None or (
            end_yaw_error is not None and abs(end_yaw_error) < yaw_tol
        )
        if not valid:
            status, reason = "unknown", "invalid_observation"
        elif terminated or truncated:
            status, reason = "unknown", "episode_ended"
        elif position_done and yaw_done:
            status, reason = "completed", "target_reached"
        elif executed_steps == 0:
            status, reason = "unknown", "no_actions_executed"
        else:
            position_progress = start_error - end_error
            yaw_progress = (
                abs(start_yaw_error) - abs(end_yaw_error)
                if target_yaw is not None
                else 0.0
            )
            position_regressed = position_progress < -threshold
            yaw_regressed = (
                target_yaw is not None and yaw_progress < -self.yaw_progress_rad
            )
            improved = (not position_done and position_progress > threshold) or (
                not yaw_done and yaw_progress > self.yaw_progress_rad
            )
            if position_regressed or yaw_regressed:
                status, reason = "no_progress", "regressed"
            elif improved:
                status, reason = "partial_progress", "error_reduced"
            else:
                status, reason = "no_progress", "stalled"

        conditions = (gripper, step_clip, action_scale, tol, yaw_tol, yaw_step_clip)
        comparable = False
        if self._reference is not None:
            reference_target, reference_yaw, reference_conditions = self._reference
            yaw_same = target_yaw is None and reference_yaw is None
            if target_yaw is not None and reference_yaw is not None:
                delta = (target_yaw - reference_yaw + np.pi) % (2 * np.pi) - np.pi
                yaw_same = abs(delta) <= yaw_tol
            comparable = (
                np.linalg.norm(target - reference_target) <= self.target_tolerance_m
                and yaw_same
                and conditions == reference_conditions
            )
        if status == "no_progress":
            if not comparable:
                self._reference = (target.copy(), target_yaw, conditions)
                self._streak = 0
            self._streak += 1
        else:
            self.reset()
        result = {
            "status": status,
            "reason": reason,
            "start_error_m": start_error,
            "end_error_m": end_error,
            "position_tolerance_m": tol,
            "progress_threshold_m": threshold,
            "executed_steps": executed_steps,
            "no_progress_streak": self._streak,
            "change_required": self._streak >= 3,
        }
        if target_yaw is not None:
            result.update(
                start_yaw_error_rad=start_yaw_error,
                end_yaw_error_rad=end_yaw_error,
                yaw_tolerance_rad=yaw_tol,
            )
        if result["change_required"]:
            result["guidance"] = (
                "Three comparable move_to calls made no meaningful progress. "
                "Re-observe or change the approach before repeating; "
                "this is EEF motion feedback, not a task verdict."
            )
        return result


def render_brief(state: Any) -> str:
    """Render at most four lines from recorded facts, without a model request."""
    records = state.records()
    if not records:
        return "[Brief]\nCurrent: no recorded observation."
    latest = records[-1]
    actions = [record for record in records if record.command]
    recent = actions[-2:]
    labels = [f"s{r.step_idx} {r.command['action']}" for r in recent]
    result = latest.result or {}
    motion = result.get("motion_assessment") or {}
    current = latest.state or {}
    xyz = current.get("robot0_eef_pos")
    grip = current.get("robot0_gripper_qpos")
    measurements = []
    if xyz is not None:
        measurements.append("eef=" + ",".join(f"{float(v):.3f}" for v in xyz) + "m")
    if grip is not None:
        measurements.append(
            f"gripper_gap_proxy={sum(abs(float(v)) for v in grip):.4f}m"
        )
    measurements.append(f"terminated={str(latest.terminated).lower()}")
    if latest.truncated:
        measurements.append("truncated=true")
    attention = "no program-confirmed issue; object holding not inferred."
    if latest.terminated:
        attention = "native environment success reported."
    elif latest.truncated:
        attention = "episode truncated; do not infer success."
    elif result.get("error"):
        attention = "tool error; inspect the original result before continuing."
    elif motion:
        attention = (
            f"move={motion['status']}; error={motion['end_error_m']:.4f}m; "
            f"tol={motion['position_tolerance_m']:.4f}m; "
            f"no_progress={motion['no_progress_streak']}/3."
        )
        if motion["change_required"]:
            attention += " Re-observe or change approach before repeating."
    elif (latest.command or {}).get("action") == "pi0_pick":
        diagnostics = result.get("diagnostics") or {}
        attention = f"pick_heuristic={str(result.get('success', 'unknown')).lower()}"
        if "descent_done" in diagnostics:
            attention += f"; descent_gate={str(diagnostics['descent_done']).lower()}"
        attention += "; holding unverified—inspect images."
    return "\n".join(
        [
            f"[Brief | step {latest.step_idx}]",
            "Recent: " + ("; ".join(labels) if labels else "initial observation") + ".",
            "Current: " + "; ".join(measurements) + ".",
            "Attention: " + attention,
        ]
    )
