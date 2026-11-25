---
id: basket-floor-localization
scope: global
kind: perception
title: Localize the visible container floor before planning an insertion.
applies_when: A clipped or lined container exposes inner walls that can bias a destination
  estimate.
symptom:
- rim-biased target
- clipped basket
- incorrect insertion center
evidence:
  cells:
  - 10_swap_t7_s0
  attempts: 1
  steps:
  - 0
  - 1
  - 8
  - 9
  - 10
  - 11
  - 19
confidence: single-shot
related:
- inspect-heuristic-negative-grasps
---

Back-project a wrist-visible floor region instead of treating visible wall samples as the cavity center.

**Why:** Depth returns the surface under each pixel. In this run, agentview samples on the white lining were on the inner wall, approximately 4.8 cm away in x and 7.2 cm higher than the wrist floor center. A floor-region estimate supported successful insertion. This is surface-selection geometry, not a camera-frame discrepancy.
**How to apply:** Identify the container in global RGB. Approach above it with clearance. Confirm the whole opening in wrist RGB, choose a region firmly on the floor, and back_project that region. Use its xy midpoint for alignment and its z for floor height. Inspect after release for settling. The tested wrist window was 150 by 215 pixels in a 1024 image; window dimensions must follow the visible floor, not be reused. Re-localize after contact if the container moves.
**Falsify:** If the chosen region demonstrably lies on the floor yet its world center consistently places objects against a wall, the region or center model is insufficient; inspect full cavity geometry.
**Related:** [[inspect-heuristic-negative-grasps]]
