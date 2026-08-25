# Day 2 — 24-pair raw RoboTwin semantic Trace audit

Status: **diagnostic completed; no correspondence GO.**

## Scope

This run is intentionally not a Motus experiment.  It uses frozen Trace on
raw, same-camera RoboTwin HDF5 images only.  It does not generate the 96-pair
training/feature cache, extract Qwen/Motus features, train an adapter, or alter
evaluation.

Inputs and selection:

```text
clean raw RoboTwin HDF5
tasks:       click_alarmclock, beat_block_hammer, open_microwave
episodes:    0, 1
cameras:     two HEAD phases, one LEFT_WRIST, one RIGHT_WRIST per episode
k:           3 frames
total:       3 tasks × 2 episodes × 4 same-camera phases = 24 pairs
```

The manifest, sparse per-query NPZ files, correspondence figures, diagnostics
and offline review page are on the training host:

```text
/kpfs-intern/binz/motus-delta-adapter/outputs/day2_semantic_trace_audit_clean_v2/
```

The Trace checkpoint hash is
`e62713636ea1d3a114b96c5785f5c7168e71a12c19dd8967d38d148950566743`.

## Execution verification

The audit code has CPU tests for direct CLI execution, deterministic 24-pair
manifest construction, same-camera/k=3 contract, sparse cycle units, zero
cycle preservation, and the offline annotation page.

```text
14 passed in 8.02s
```

The A800 run completed all 24 records without NaN/OOM.  Peak Trace allocation
was 3.31 GiB.  The only Trace warning was the existing uncompiled RoPE2D
fallback, which affects latency rather than this frozen prediction's result.

## Numerical diagnostics

All 1,152 sparse queries were finite.  This is only a geometry-completeness
fact; it is **not** a validity or visibility claim.

| Camera | Query count | Motion median | Cycle median / P95 | Median d1/d2 | d1/d2 >= 0.98 | Cycle >= 3px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HEAD | 576 | 2.80px | 0.00 / 13.06px | 0.9352 | 26.2% | 16.1% |
| LEFT_WRIST | 288 | 6.73px | 1.77 / 20.74px | 0.9847 | 54.5% | 39.6% |
| RIGHT_WRIST | 288 | 7.07px | 2.65 / 25.07px | 0.9863 | 61.1% | 40.3% |
| All | 1,152 | — | — | — | 42.0% | 28.0% |

`d1/d2` close to one means the nearest and second-nearest 3-D descriptor
candidates are almost tied, i.e. a high ambiguity signal.  The wrist cameras
are therefore the most concerning setting: they contain both more motion and
substantially more ambiguous / cycle-inconsistent sparse matches.

## Visual inspection

I inspected all three 8-pair correspondence contact sheets plus high-motion
wrist examples.  At this 64px sparse query resolution, the fields appear
globally coherent and no obvious *large-scale* arm-to-table/background collapse
was observed.  This is only a weak negative observation.

It is **not** a positive foreground result: many manipulated objects occupy a
small region or fall between the uniform 64px query lines, so this audit cannot
quantify object correspondence quality.  The machine-generated review page is
specifically retained for human semantic labels (`2/1/0/X`) rather than treating
my coarse visual scan as ground truth.

## Decision

Combined with the earlier synthetic +32px calibration result (median 12px EPE,
with both target-time point NN and full-control-trajectory NN), Step 0 does not
support a dense physical correspondence operator suitable for feature warp.

```text
Dense 96-pair Trace cache:         do not build
Qwen fixed-vs-aligned comparison:  do not run yet
Delta Adapter training:            do not start
```

The most defensible statement is narrower: frozen Trace can supply a
coarse, same-camera diagnostic field, but finite output and visually smooth
motion do not establish token-level physical correspondence—particularly under
wrist-camera motion.

Before any later GO decision, use a separately reviewed, foreground-targeted
point audit (rather than only a uniform sparse grid) and/or replace the
Trace-to-2-D matching derivation.  Do not relax this gate based only on the
100% finite fraction.
