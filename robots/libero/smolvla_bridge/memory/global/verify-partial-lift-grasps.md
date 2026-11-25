---
id: verify-partial-lift-grasps
scope: global
kind: primitive
title: Verify a closed grasp when the lift threshold is unmet.
applies_when: Pi0 stops with partially closed fingers and a target at the gripper
  but reports success false after a small ascent.
symptom:
- pi0-success-false
- partial-lift
- neck-grasp
- closed-fingers
evidence:
  cells:
  - goal_swap_t9_s0
  attempts: 2
  solved_seeds:
  - 0
confidence: single-shot
related:
- target-support-interiors
---


A Pi0 lift-threshold failure can coexist with a real grasp; verify it visually before retrying the policy.

**Why:** The tool's heuristic requires sufficient post-descent ascent. In the solved A3 call, ascent was .01807 m below the .05 m threshold, while finger separation was .02167 m and the neck was between the fingers. A subsequent closed scripted lift raised the same bottle and emptied its source spot.

**How to apply:** Inspect the target at the gripper and nonzero finger separation. A3 used +1 for 8 steps to firm, then a slow vertical lift at step_clip .01, holding +1. Compare both camera views before carrying laterally. The .0167-.0217 m openings observed here were neck-specific, not universal object thresholds. If uncertain, use a small verification lift before a full carry.

**Falsify:** The target stays on its support while the gripper rises, falls, or the fingers fully close on air. In that case do not treat this as a confirmed grasp.

**Related:** [[target-support-interiors]]
