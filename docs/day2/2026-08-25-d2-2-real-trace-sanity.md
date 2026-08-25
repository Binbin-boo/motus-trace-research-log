# Day 2.2 — frozen Trace real-image sanity result

Status: **executed once; partial pass only.** This does not yet justify running
a large RoboTwin correspondence audit.

## Command

```bash
cd /kpfs-intern/binz/motus-delta-adapter
.venv/bin/python tools/run_trace_sanity.py \
  --camera head --frame-index 24 \
  --translation-dx 32 --translation-dy 0 --device cuda:0 \
  --output-dir outputs/day2_trace_sanity_clean_head_ep0_f24
```

The input is raw clean `click_alarmclock` `episode0.hdf5`, HEAD camera, frame
24 (`640 x 480`). The synthetic pair is `I_cur = shift(I_prev, +32, 0)` in
original camera pixels, so the required backward mapping is `(-32, 0)` in the
same coordinate system.

## Exact observed Trace contract

```text
Trace transform: original 640 x 480 -> 512 x 384
transpose: false
crop: none
control points: 10
ctrl_pts3d: [10, 384, 512, 3]
ctrl_conf:  [10, 384, 512]
track_pts3d target field: [1, 384, 512, 3]
track_conf target field:  [1, 384, 512]
```

The predicted input-view times were approximately `0.0003` and `0.9999`,
consistent with the ordered `previous=0`, `current=1` input contract.

## Metrics (192 sparse Trace queries, stride 32)

| Pair | finite fraction | backward dx median | displacement median / p95 | endpoint error median / p95 |
| --- | ---: | ---: | ---: | ---: |
| identity | 1.000 | `0.00 px` | `1.77 / 4.51 px` | `1.77 / 4.51 px` |
| +32 px image shift | 1.000 | `-21.25 px` | `21.29 / 38.25 px` | `12.00 / 17.72 px` |

The sign is correct: the model maps a right-shifted current image back to the
left in the previous image. However, its magnitude is materially smaller than
the expected `-32 px`; the translation median endpoint error is `12 px`.

## Runtime

```text
identity forward:    0.575 s, 3.310 GiB peak allocated GPU memory
translation forward: 0.167 s, 3.311 GiB peak allocated GPU memory
```

The environment warned that CUDA-compiled RoPE2D is unavailable and used a
slower PyTorch implementation. This affects speed profiling, not the output
geometry result. No NaN or out-of-memory event occurred.

## Outputs retained on the remote host

```text
/kpfs-intern/binz/motus-delta-adapter/outputs/day2_trace_sanity_clean_head_ep0_f24/
  previous.png
  identity_current.png
  translated_current.png
  metrics.json
```

## Decision after D2.2

**Do not proceed to 100 real temporal pairs yet.** The basic pipeline is
operational and the backward direction is not inverted, but the synthetic
translation is not accurate enough to call the derived correspondence reliable.

The next investigation must distinguish these possibilities before changing
the method:

1. an audit-script coordinate/metric issue;
2. expected limitation of 3-D trajectory NN on an image-wide artificial shift;
3. a genuine Trace robustness failure on this camera/image distribution.

No Motus code, policy, or training run was changed by this result.
