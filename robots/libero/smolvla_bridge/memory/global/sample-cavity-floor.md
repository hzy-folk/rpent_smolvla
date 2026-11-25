---
id: sample-cavity-floor
scope: global
kind: perception
title: Refine lined containers using their visible inner floor.
applies_when: Agentview container samples strike a raised inner wall or lining.
symptom:
- rim-bias
- lining-bias
- cavity-center
evidence:
  cells:
  - object_task_t5_s0
  steps:
  - 0
  - 1
confidence: single-shot
related: []
---

Agentview lining samples can differ materially from a wrist-localized floor center.

**Why:** Back-projection measures the sampled surface; the lining is a different surface from the floor. Here median lining z was 0.1057 m and floor-region z was 0.0139 m, with a 5.3 cm x difference.
**How to apply:** Identify the container in agentview, move above it, and select a wrist region wholly inside the visible floor. Use a depth band to exclude walls; this run used floor z through floor+0.05 m. Do not transfer its absolute coordinates to other scenes. The resulting point was consistent with the final occupied cavity, but a scripted placement at that point was not tested.
**Falsify:** A floor-bounded region is systematically displaced from the visually centered cavity despite correct camera and depth alignment.
**Related:** None.
