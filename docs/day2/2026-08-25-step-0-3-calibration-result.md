# Day 2 Step 0.3 — coordinate calibration result

Status: **code geometry passed; neither tested Trace-to-2-D matching operator is pixel-calibrated.**

## New CPU checks

Remote command:

```bash
cd /kpfs-intern/binz/motus-delta-adapter
.venv/bin/python -m pytest -q tests/test_run_trace_sanity.py tests/test_trace_pair.py
```

Result:

```text
12 passed in 4.25s
```

The tests verify exact raw-camera transform round trips for both actual source
sizes (`640 x 480`, `320 x 240`), correct `+32px current -> -32px previous`
oracle direction, auditable NPZ output, and overlay generation. The 12px model
EPE therefore cannot be explained by an x/y, resize, crop, or backward-sign bug
in the audit script.

## Frozen Trace r2 result

The same one-frame clean HEAD experiment was rerun after adding diagnostics.
The numerical result is unchanged, as expected: no model/preprocessing/matching
algorithm was changed.

```text
Identity:          median displacement 1.77px, P95 4.51px
Synthetic +32px:   backward dx median -21.25px (required -32px)
Synthetic +32px:   endpoint-error median 12.00px, P95 17.72px
Finite fraction:   1.00
```

## Per-query inspection

Remote output:

```text
/kpfs-intern/binz/motus-delta-adapter/outputs/day2_trace_sanity_clean_head_ep0_f24_r2/
  metrics.json
  identity_queries.npz
  translation_queries.npz
  identity_correspondence_overlay.png
  translation_correspondence_overlay.png
```

The overlay and top-EPE points show that errors are not a single global
coordinate offset. In low-texture table/background regions, the match often
under-travels horizontally and sometimes drifts vertically. Example: a current
point at `(40.12, 360.12)` should map to `(8.12, 360.12)` but was matched to
`(30.12, 348.88)`, EPE `24.71px`.

The NN ambiguity diagnostic is also high:

```text
translation median d1/d2: 0.9807
translation P95 d1/d2:    0.9995
```

Values close to one mean the nearest and second-nearest previous 3-D points are
nearly tied. This is consistent with a target-time point-only NN ambiguity on
flat or repetitive regions. It is not an occlusion probability and does not by
itself prove that Trace trajectories are unusable on visible robot/object areas.

## Full-trajectory cross-check (r3)

The planned C2-style cross-check was then run on the same frozen RoboTwin
HEAD frame and synthetic shift.  It does not change Trace, Motus, input
preprocessing, or the point-NN result.  It only replaces the descriptor used
for NN lookup:

```text
point NN:             3-D location at the previous target time
trajectory NN:        all 10 Trace 3-D control points concatenated
```

The CPU test suite was extended first and passed:

```text
14 passed in 3.09s
```

The actual A800 run produced:

| Metric | Target-time point NN | Full 10-control-point trajectory NN |
| --- | ---: | ---: |
| Identity median displacement | 1.77px | 1.77px |
| Synthetic +32px backward dx median | -21.25px | -21.25px |
| Synthetic translation EPE median | 12.00px | 12.00px |
| Synthetic translation EPE P95 | 17.72px | 17.46px |
| Translation median d1/d2 | 0.9807 | 0.9816 |

Thus the full descriptor does **not** resolve the under-travel or the
near-tied-neighbour ambiguity.  The new overlay retains coherent horizontal
motion but shows the same non-global, low-texture/background matching error.
It is therefore not defensible to choose the full-trajectory version merely
because it is theoretically closer to Trace's trajectory consistency idea.

Remote r3 artifacts:

```text
/kpfs-intern/binz/motus-delta-adapter/outputs/day2_trace_sanity_clean_head_ep0_f24_r3/
  metrics.json
  translation_correspondence_overlay.png
  translation_full_trajectory_overlay.png
  translation_queries.npz
  translation_full_trajectory_queries.npz
```

The run used a raw RoboTwin clean HDF5 camera frame, not an auxiliary dataset:

```text
/kpfs-intern/datasets/worldarena2/robotwin_full/click_alarmclock/
  aloha-agilex_clean_50/data/episode0.hdf5
```

Trace inference itself is healthy: fields are finite, correct shapes were
returned, and peak allocated memory was 3.31 GiB.  The failure is specifically
the proposed dense 3-D-NN-to-2-D correspondence derivation.

## Decision

**Step 0 is not a GO.**  The direction and coordinate transforms are correct,
but the two available NN derivations both miss a known 32px displacement by a
median 12px.  Do not call either a validated pixel correspondence operator and
do not use it as the accepted 96-pair / Qwen-warp cache.

The safe next Day-2 action is a *small, explicitly diagnostic* real-RoboTwin
semantic audit (20--30 pairs), which can establish whether the error is mostly
in low-texture background or also produces arm/object-to-background jumps.  It
must be reported as an audit only, not as the feature-alignment experiment. If
that audit shows systematic object/arm errors, Day-2 is NO-GO. If it is
visible-region usable only, a separately reviewed reliability/matching remedy
is still required before the 96-pair cache and frozen-Qwen comparison.

No Motus feature, training, policy, or evaluation code was changed.
