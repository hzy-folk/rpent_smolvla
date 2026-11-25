---
id: task-family_libero_object_task_t8
scope: task-family
suite: libero_object
regime: task
task_id: 8
task_language: Pick the salad dressing and place it in the basket
confidence: single-shot
---

# Salad dressing: grasp, verify, and place

Identify the green-capped Ranch Dressing bottle visually. Find the basket cavity using its visible floor rather than the wall or lining. Recompute coordinates in this episode.

The historical SmolVLA seed-0 success needed four bounded pi0_pick calls. After three unsuccessful calls, the planner re-localized, reopened/repositioned the gripper, returned above the bottle, and lowered the approach to about 0.22 m in that scene before the fourth call. Do not treat that height or its xy as valid without current observations.

A false pick heuristic did not establish grasp failure: the fourth call's measured descent failed its gate, but images showed the bottle lifted with a gripper opening around 0.0366 m. Confirm object motion in images before carrying. An empty closed gripper is not evidence of a grasp.

After visual confirmation, the planner clamped for 10 steps, lifted vertically, used an intermediate waypoint with the gripper closed, centered over the basket cavity, and released. The native environment reported terminated=true on release.

Use short grasp calls (historical max_chunks=8) and inspect the result. If grasp is not confirmed, re-localize and change approach rather than carrying empty fingers. Confirm continued retention after movement and require native termination to declare success.

The bridge instruction stays fixed to the training phrase; changing a high-level tool prompt does not test another SmolVLA instruction. This record describes one success, and other seed-0 runs failed. It is not a deterministic replay or a reliability estimate.
