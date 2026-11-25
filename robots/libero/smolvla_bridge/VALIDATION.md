# Validation record

Date: 2026-11-25. Base RPent revision: `b96cbc15d9ba73cf47dc63e3c2512d423bae5365`, plus the local working changes. The existing optional episode features and unrelated workspace changes were preserved. The external bridge directory was not modified.

## Checks completed

- Full offline suite: `pytest tests/unit_tests -q` — **2141 passed, 13 skipped**. Three warnings concern the existing environment's missing pytest-timeout plugin/mark registration.
- Ruff preview lint and formatting across `rpent robots tests`, `git diff --check`, and shell syntax checks passed.
- English and Chinese Sphinx builds passed with `-W --keep-going`.
- Mock CLI runs from another working directory without model assets, returning `(1, 5, 7)`.
- The actual RPent Pi05VLAClient successfully calls a standalone mock server over loopback HTTP, receives `(5, 7)` float32 actions, and receives an acknowledged shutdown before process exit. The migrated server waits for response-handler threads during shutdown.
- Adapter tests preserve the training instruction regardless of planner text, image orientation/layout/range, and action slicing/bounds.
- Portable profile tests cover environment-variable expansion and early errors for missing processor assets.
- The published LeRobot 0.3.3 wheel's modeling_smolvla.py, configuration_smolvla.py, and smolvlm_with_expert.py are byte-identical to the development source. Wheel SHA256: `ae8354b516f5c8cfaa059d5ac926b4d03e96ca73350b971f8c7d75b905ba953d`.
- Real checkpoint inference passed on the local RTX 5060 Laptop GPU using the migrated bridge, published LeRobot 0.3.3 source, existing model environment, and existing assets. A temporary compatibility profile supplied the source and supplemental dependency paths; those machine paths are not part of the published profile. Model loading took about 18 seconds and synthetic-observation inference about 0.62 seconds. Result: `(1, 5, 7)` float32.
- The documented isolated requirements resolved successfully with uv's `--dry-run --torch-backend cu128`; no existing environment was changed.
- All ten local asset files matched `assets_manifest.json`. The pinned public base configuration matched. Five public processor files matched directly; the other two differ only in chat-template fields, and their training versions are included as processor_overrides.
- A wheel build includes profiles, processor overrides, scripts, manifests, guides, and reference memory. It excludes private env.local.sh and model weights. The extracted bridge also passed the mock CLI check outside the source checkout.

## Limits

No full fresh-machine installation was performed. No new paid planner/API episode was run for this migration; earlier feature-on/off seed-0 tests are separate evidence. A synthetic-observation model call does not prove task success. The new minimal memory corpus differs from the full development corpus, and no benchmark success rate is claimed.

The fine-tuned checkpoint has no public download URL yet. Publish the exact file separately and update checkpoint_download_url before describing the real task as reproducible from a public download alone. The manifest already records its identity without including its bytes.
