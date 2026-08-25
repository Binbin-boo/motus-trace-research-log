# Day 2 Step 0.3 — coordinate calibration result

Status: **code geometry passed; target-time single-point 3-D NN remains uncalibrated.**

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

## Decision

Do not build the 96-pair cache with target-time point-only NN as the accepted
correspondence operator. The next small diagnostic is the planned C2-consistent
cross-check: match the full 10-control-point trajectory descriptor, compare it
with target-time NN on the same identity/+32px pair, and inspect whether it
reduces ambiguity/EPE without changing Trace or Motus.

No Motus feature, training, policy, or evaluation code was changed.
