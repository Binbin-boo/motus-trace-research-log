# Day 2.1 — synthetic correspondence geometry tests

Status: **passed on 2026-08-25.**  This stage did not load Trace Anything,
Motus, Qwen, WAN, or a GPU.

## Files under test

Remote canonical project paths:

```text
/kpfs-intern/binz/motus-delta-adapter/tools/trace_pair.py
/kpfs-intern/binz/motus-delta-adapter/tests/test_trace_pair.py
```

They are intentionally isolated from model/training imports.  The module only
accepts target-time 3-D point fields and derives explicit 2-D NN mappings.

## Command

```bash
cd /kpfs-intern/binz/motus-delta-adapter
.venv/bin/python -m pytest -q tests/test_trace_pair.py
```

## Result

```text
......                                                                   [100%]
6 passed in 1.61s
```

## Covered contracts

| Test | Contract established |
| --- | --- |
| identity point fields | backward mapping is identity and uses `(x, y)` order |
| known translation | current-to-previous displacement has the expected negative sign |
| reverse + cycle | a consistent translated field returns to the original current pixel |
| non-finite 3-D query | it remains invalid; finite candidate points cannot make it valid |
| reliability rule | only finite, finite-cycle, below-threshold points are reliable |
| camera validation | head/wrist cross-camera input is rejected |

## Interpretation

This is a unit-level geometry result only.  It confirms that later failures in
the real audit are not silently caused by an x/y swap, the basic backward
direction, an all-NN-is-valid rule, or an untested cycle composition.  It says
nothing yet about the quality of Trace Anything trajectories on RoboTwin.

## Next review gate

Propose D2.2: add no Motus code, but call frozen Trace directly on one real
same-camera identity pair and one synthetic translated RoboTwin image.  Record
the actual Trace input transform, identity displacement, translation endpoint
error, time, and peak GPU memory before attempting a real temporal pair.
