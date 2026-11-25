---
id: verify-carry-retention
scope: global
kind: failure
title: Verify retention after every carry motion.
applies_when: Carrying an object with a continuously closed gripper command.
symptom:
- slip
- empty-gripper
- carry
- shrinking-gap
evidence:
  cells:
  - goal_swap_t2_s0
  steps:
  - 8
  - 9
  - 10
confidence: single-shot
related: []
---

A +1 gripper command does not establish that an object stayed in the fingers during motion.

**Why:** A commanded closure is an action, not a contact observation. Here the held bottle visibly followed two lifts, then left the fingers during a +1 translation; the precise slip cause is unknown.
**How to apply:** Compare wrist RGB and finger separation after every primitive. This run's opening decreased from approximately 0.0495 m after the first lift to 0.0415 m after the second, then 0.0054 m after loss. Use the near-empty gap together with visual departure, not a universal numeric threshold alone. Stop carrying when loss is confirmed and localize the object again unless official termination has already fired.
**Falsify:** A tool result showing the object still between the fingers despite the inferred loss would invalidate that diagnosis; small gaps alone are insufficient.
**Related:** None.
