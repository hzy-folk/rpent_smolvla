---
id: inspect-heuristic-negative-grasps
scope: global
kind: primitive
title: Inspect the held object when a pick heuristic reports failure.
applies_when: Short Pi0 calls split descent and ascent or the held object keeps the
  fingers wider than the success threshold.
symptom:
- success false
- object visibly lifted
- descent_done false
- wide grasp
evidence:
  cells:
  - 10_swap_t7_s0
  attempts: 1
  steps:
  - 5
  - 6
  - 14
  - 15
  - 19
confidence: single-shot
related:
- basket-floor-localization
---

A negative pick heuristic can coexist with a secure, task-solving grasp.

**Why:** Returned diagnostics on continuation calls showed ascent without a new descent, so descent_done remained false. The can also held the fingers ~0.06249 m apart, above the configured 0.06 threshold. The box lifted 0.2695 m and the can 0.1003 m in their continuation calls; both were visually held and subsequently carried successfully.
**How to apply:** Use grasp-only prompts and inspect after short calls (8 chunks for approach in this run; continuations used 3–8). Confirm the same target raised between the fingers and an empty source location, with a nonzero gripper gap. Once confirmed, switch to scripted +1 carry rather than extending Pi0 just to obtain success:true. Box firming used 10 steps; can 5. These are observed settings, not an experimentally established range.
**Falsify:** If the apparently raised object stays behind on the first small closed-gripper lift/carry, visual evidence was insufficient; re-localize and regrasp. A false flag alone never establishes that the grasp is secure.
**Related:** [[basket-floor-localization]]
