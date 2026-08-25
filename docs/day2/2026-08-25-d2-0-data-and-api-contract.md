# Day 2.0 — per-camera data and Trace API contract

Status: **completed read-only on 2026-08-25.**  No model was loaded, no GPU work
was run, and no Motus/Trace source file was changed.

## Decision

Day-2 can use genuine same-camera pairs.  It must read them from the raw HDF5
episodes, not crop the processed Motus composite MP4.

## Raw RoboTwin sources

| Split | Raw root (task-relative) | Episodes for `click_alarmclock` | One camera frame |
| --- | --- | ---: | --- |
| clean | `/kpfs-intern/datasets/worldarena2/robotwin_full/<task>/aloha-agilex_clean_50/data` | 50 | `640 x 480` (W x H) |
| randomized | `/kpfs-intern/datasets/worldarena2/robotwin_rand_full/<task>/aloha-agilex_randomized_500/data` | 500 | `320 x 240` (W x H) |

Each HDF5 file contains JPEG-byte arrays, with equal frame counts for the
three views.  Exact keys are:

```text
HEAD         observation/head_camera/rgb
LEFT_WRIST   observation/left_camera/rgb
RIGHT_WRIST  observation/right_camera/rgb
```

All three actual views are landscape.  Consequently the official inference
loader does not apply its portrait rotation to these raw frames.

## Verified link to Motus processed data

For clean `click_alarmclock`, raw `episode0.hdf5` has 77 frames per camera and
processed `clean/click_alarmclock/videos/0.mp4` has 77 frames at `320 x 360`.

Recreating the converter's T-layout from raw frame 0, then resizing it to
`320 x 360`, produces the processed video frame up to expected JPEG/MP4 codec
loss:

```text
mean absolute RGB difference: 4.6340 / 255
99th-percentile absolute difference: 18 / 255
fraction of channel values within 8: 0.8392
```

This establishes the episode and frame-index convention needed for the audit.
It does not claim raw and decoded-MP4 pixels are byte-identical.

## Actual concatenation contract

The RoboTwin converter reads original head/left/right frames, resizes each
wrist image to half the head width and height, concatenates wrist views
horizontally below the head view, then resizes the result to `320 x 360` for
the MP4.  Thus the composite is correct as a Motus input but must not be a
single camera plane for Trace.

## Installed Trace Anything contract

Installed source (vendored, no nested git repository):

```text
/kpfs-intern/binz/motus-delta-adapter/third_party/TraceAnything
trace_anything.py SHA-256: 854538ed70fbac56d2a24e81e37da84fcf748af9ea4edf67de88017713ab41c1
infer.py SHA-256:           8aa565fc63f07c14159b12fa98e161e878161ea1572881b0a7a2e3d38524a9df
```

The supplied `infer.py` preprocessing is exactly:

```text
if H > W: rotate portrait by 90 degrees
resize the long side to 512 pixels
crop bottom/right to a common multiple of 16
normalise RGB to [-1, 1]
```

Therefore its expected Trace input size for both raw camera resolutions is
`512 x 384` (W x H), with a single resize and no rotation/crop.  The Day-2
implementation must still record this transform as metadata rather than rely
on this special case.

The configuration uses `poly_degree: 10`, `landscape_only: true`, and
`conf_mode: [exp, 1, inf]`.  Its confidence is therefore a positive relative
score with lower bound 1, not a probability.

### `model.forward(views)` input

`views` is an ordered list.  Each element contains:

```python
{"img": Tensor[1, 3, H, W], "time_step": float}
```

For a two-frame backward pair, the planned unambiguous order is:

```text
views[0] = previous frame, time_step = 0.0
views[1] = current  frame, time_step = 1.0
```

### `model.forward(views)` return

It returns one dictionary per source image.  With two views:

```text
results[0]  source pixels in previous frame
results[1]  source pixels in current frame

ctrl_pts3d  [K, H, W, 3]
ctrl_conf   [K, H, W]
track_pts3d list of two [1, H, W, 3] tensors
track_conf  list of two [1, H, W] tensors
```

The target index in each `track_*` list follows the input-view order.  Hence
the Day-2 primary fields are:

```text
current at previous time = results[1]["track_pts3d"][0]
previous at previous time = results[0]["track_pts3d"][0]
```

Their 3-D nearest-neighbour match derives `current -> previous` pixel
correspondence.  The reverse pairing uses `results[0]["track_pts3d"][1]` and
`results[1]["track_pts3d"][1]`.

The supplied official saving script explicitly removes `track_pts3d` and
`track_conf` before writing `output.pt`.  Day-2 must call the frozen model
directly and consume these fields in memory.

## Motion proxy, not occlusion

The supplied `fg_mask` is created outside `model.forward`: variance across
`ctrl_pts3d` control points, followed by Otsu-style thresholding.  It is a
motion/dynamic proxy only.  Day-2 will name it `motion_proxy`, never
`occlusion_mask`.

## Next review gate: D2.1

The next proposed patch is limited to new isolated files:

```text
tools/trace_pair.py
tests/test_trace_pair.py
```

It will first implement pure coordinate/nearest-neighbour/cycle helpers with
synthetic tests only.  It will not load Trace, Motus, Qwen, or a GPU until that
patch and its intended tests have been reviewed.
