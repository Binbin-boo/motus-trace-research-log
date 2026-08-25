# Day 2 — revised correspondence-alignment mechanism plan

> **For agentic workers:** execute inline and stop after every code patch, test run, and result for user review.

**Goal:** Determine whether physical correspondence makes frozen Motus visual representations less motion-dependent than fixed-grid comparison, without training any adapter.

**Architecture:** Trace is a frozen measurement tool. It provides same-camera current-to-previous coordinate hypotheses and reliability diagnostics. Those coordinates are mapped into the canonical Motus composite/Qwen grid, then used to compare fixed-grid and correspondence-aligned discrepancies on identical frozen tokens.

**Spec:** `docs/day2/2026-08-25-trace-correspondence-plan.md` plus the reviewed Step 0–12 protocol.

## Non-negotiable constraints

- Do not train, modify Motus/Qwen/WAN weights, add an adapter, or change evaluation.
- Trace calls are same-camera only; raw HDF5 is the sole Trace input.
- `tools/trace_pair.py` remains geometry-only and has no Motus/Qwen/WAN/model-loader import.
- Preserve track confidence, d1, d1/d2, cycle error, and motion proxy separately.
- `finite` is not visibility; motion proxy is not an occlusion mask.
- Use the same frozen checkpoint, preprocessing, pair manifest, and accepted token set in fixed, aligned, and control analyses.
- Start with camera-pure Qwen visual tokens; reject panel-boundary/padding tokens.
- Generated data stays in `outputs/day2_*`; do not commit images/caches/checkpoints.

## Work already completed

| Completed check | Result | Meaning |
| --- | --- | --- |
| D2.0 raw camera + Trace API audit | raw HDF5 found; 3 cameras verified; fields/shapes verified | per-camera audit is feasible |
| D2.1 synthetic geometry unit tests | 6 passed | direction, NN, cycle, finite/reliability semantics covered |
| D2.2 frozen Trace identity/translation | identity median 1.77px; +32px shift median -21.25px; EPE median 12px | backward sign is correct, but calibration is not yet acceptable |

The 12px synthetic EPE keeps Step 0 **open**. We do not create a 96-pair cache or extract Motus features before it is diagnosed.

## Review-gated steps

### Step 0.3 — close the Trace calibration gate

**Question:** Is the 12px EPE caused by our geometry/metric code, or by Trace+3D-NN itself?

**Files:** create `tests/test_run_trace_sanity.py`; modify `tools/run_trace_sanity.py`.

**Tests to add before GPU execution:**

```python
@pytest.mark.parametrize("height,width", [(480, 640), (240, 320)])
def test_landscape_transform_round_trip(height, width):
    meta = TraceTransformMeta.for_shape(height, width)
    points = torch.tensor([[[0., 0.], [width - 1., height - 1.], [width * .37, height * .61]]])
    assert (meta.trace_to_original_xy(meta.original_to_trace_xy(points)) - points).abs().max() < 1e-4

def test_shift_oracle_has_negative_backward_direction():
    current = torch.tensor([[[128., 120.]]])
    assert torch.equal(current - torch.tensor([32., 0.]), torch.tensor([[[96., 120.]]]))
```

**Script diagnostics to add:** `queries.npz` containing source/oracle/matched original coordinates, EPE, d1, d1/d2, finite flag; one coloured point-overlay image. Existing `metrics.json` remains aggregate-only.

**Approved GPU command after code review:**

```bash
cd /kpfs-intern/binz/motus-delta-adapter && .venv/bin/python tools/run_trace_sanity.py --camera head --frame-index 24 --translation-dx 32 --translation-dy 0 --device cuda:0 --output-dir outputs/day2_trace_sanity_clean_head_ep0_f24_r2
```

**Gate:** round-trip error `<1e-4px`; review high-EPE overlay points; retain identity metrics. If actual Trace EPE stays `>8px`, record that it is not pixel-accurate and decide explicitly whether visible-region Conditional GO is defensible. Do not hide this result.

### Step 1 — freeze a 96-record temporal manifest

**Question:** What exact data will every later comparison use?

**Files:** create `tools/build_day2_pair_manifest.py` and `tests/test_day2_pair_manifest.py`.

Use 3 tasks × 4 episodes × 8 positions × `k=3` = 96 temporal records. Initial tasks: `click_alarmclock`, `beat_block_hammer`, `open_microwave`. Selection seed is `20260825`. Each record carries all three camera names, absolute raw HDF5 path, split/task/episode, previous index, current index, and `k`.

**Tests:** same seed produces identical JSON; `current - previous == 3`; all three camera arrays contain both indices; no duplicate `(task, episode, current)`.

**Review gate:** inspect `outputs/day2_alignment_v1/pair_manifest.json` before any cache GPU run.

### Step 2 — cache derived per-camera Trace correspondence

**Question:** Can each manifest record produce raw, auditable current-to-previous mappings?

**Files:** create `tools/build_day2_trace_cache.py` and `tests/test_day2_trace_cache_schema.py`; reuse `tools/trace_pair.py`.

Every record stores, per camera: `backward_xy_original`, `forward_xy_original`, source/target track confidence, `nn_distance`, `nn_ratio`, `cycle_error_px`, `finite_mask`, `motion_proxy`, transform metadata, manifest identity, and Trace checkpoint SHA. Save two sparse grids: 16px visual-review points and future Qwen token centres. Never save an all-pixel distance matrix.

**Tests:** required schema exists; finite coordinates remain inside their source camera; metadata/checkpoint match; cross-camera request raises an error.

**Review gate:** run exactly one temporal record (three cameras), inspect it, then approve remaining 95.

### Step 3 — apply transparent reliability and manual quality control

**Question:** Is Trace a sufficiently trustworthy ruler on visible robot scenes?

**Files:** modify `tools/build_day2_trace_cache.py`; create `tests/test_day2_reliability.py` and `tools/review_day2_trace_cache.py`.

First-pass rule is exactly:

```python
reliable = finite_mask & torch.isfinite(cycle_error_px) & (cycle_error_px < tau_cycle_px)
mismatch_candidate = finite_mask & ~reliable
```

Confidence and d1/d2 are reported, not secretly folded into this first rule. Choose `tau_cycle_px` after the one-record distribution and write it to metadata before the 96-record run.

Generate `manual_trace_review/review.html` for 20–30 representative records. Label inspected points: `2` clearly same point, `1` plausible, `0` wrong, `X` no true correspondence. Cover static, arm, manipulated object, wrist camera motion, and disocclusion candidates.

**Gate:** STOP for systematic object→arm/table jumps, incoherent wrist-camera fields, or no metric separation of obvious errors. Conditional GO permits visible-only, reliability-masked analysis.

### Step 4 — bridge raw-camera correspondence to canonical Qwen tokens

**Question:** Does a raw-camera mapping land at the correct previous feature location without crossing panels?

**Files:** create `tools/day2_camera_feature_geometry.py` and `tests/test_day2_camera_feature_geometry.py`; read `data/utils/multi_camera_concat.py` and canonical VLM preprocessing.

Provide:

```python
camera_current_token_centres_to_previous_feature_uv(
    camera, current_token_centres_qwen_xy, backward_xy_raw,
    motus_composite_meta, qwen_grid_meta,
) -> tuple[np.ndarray, np.ndarray]
```

It returns bilinear `grid_sample` UV and a camera-pure/in-bounds mask. Composite panel geometry is calculated from actual camera sizes, never hardcoded.

**Tests:** raw→composite→raw round trip `<1e-4px`; padding/boundary token rejected; accepted token stays in its own camera; identity map returns same feature coordinate.

**Review gate:** one image of HEAD/LEFT/RIGHT accepted token centres and mapped locations.

### Step 5 — extract the first representation: Qwen Vision merged grid

**Question:** Does correspondence matter before deeper multimodal mixing?

**Files:** create `tools/extract_day2_representations.py` and `tests/test_day2_representation_schema.py`.

Use frozen Qwen post-merge visual tokens from the canonical Motus composite. Previous/current must use identical canonical image and prompt preprocessing. Save selected camera-pure features, grid metadata, pair identity, and no graph/checkpoint.

**Tests:** same grid shape for both frames; Qwen has no trainable parameter; identical images create identical processor inputs; each selected token has a cache correspondence.

**Review gate:** inspect one record’s shape, dtype, grid, and selected-token count. Do not inspect deeper layers yet.

### Step 6 — fixed-grid vs correspondence-aligned discrepancy

**Question:** Does correct physical alignment reduce apparent temporal discrepancy?

**Files:** create `tools/analyze_day2_alignment.py` and `tests/test_day2_alignment_metrics.py`.

For the same reliable token set:

```python
d_fixed = 1 - cosine_similarity(V_current[i], V_previous[i])
V_previous_warped = grid_sample(V_previous, C_current_to_previous[i])
d_corr = 1 - cosine_similarity(V_current[i], V_previous_warped)
gain = (d_fixed - d_corr) / (d_fixed + 1e-8)
motion = norm(C_raw[i] - token_centre_raw[i]) / camera_diagonal
```

Report per-pair median gain, `P(d_corr < d_fixed)`, and low/middle/high-motion terciles. Aggregate per temporal record first, then paired-bootstrap across 96 records; never treat all tokens as independent episodes.

**Tests:** known feature-grid translation yields `d_corr < d_fixed`; identity yields zero; invalid UV is excluded rather than clamped; fixed/corr use identical tokens.

### Step 7 — central mechanism test: motion dependence

**Question:** Does alignment reduce the association between motion and feature discrepancy?

**Files:** modify `tools/analyze_day2_alignment.py`.

For every temporal record and its reliable token set:

```text
rho_fixed = Spearman(motion, d_fixed)
rho_corr  = Spearman(motion, d_corr)
delta_rho = rho_fixed - rho_corr
```

Report per-record values, paired-bootstrap intervals, and two scatters. Retain contact/state-change cases: a high aligned residual can be correct evidence rather than a failure.

### Step 8 — controls that test correspondence, not smoothing

**Question:** Is improvement specific to correct physical correspondence?

**Files:** modify `tools/analyze_day2_alignment.py` and its metric tests.

On the same valid tokens, compare:

```text
correct_trace:   current -> previous Trace UV
wrong_direction: previous -> current UV used as backward
random_same_cam: seed-20260825 permutation of valid previous UV within camera/pair
self_warp:       current feature sampled at own identity UV (sampler sanity only)
```

Random never crosses cameras/padding. Self-warp must reproduce current features within floating tolerance but is not a scientific baseline. Correct Trace must beat random/wrong direction; otherwise the result is interpolation/smoothing, not correspondence evidence.

### Step 9 — mechanism visualisation and optional layers

**Files:** create `tools/render_day2_alignment_review.py`; modify extraction/analysis only after Step 6–8 are clear.

Render previous/current/correspondence, motion, fixed discrepancy, aligned discrepancy, gain, reliable mask, and residual tokens. Include rigid movement and interaction/state-change cases with manifest IDs. Only after Vision merged evidence is control-specific, audit Qwen final visual positions, then Motus VLM-adapter output with read-only hooks. No favorable-layer search if Vision evidence fails.

### Step 10 — final decision

Create `docs/day2/YYYY-MM-DD-day2-mechanism-result.md` and `outputs/day2_alignment_v1/analysis/final_summary.json`.

- **GO:** acceptable Trace audit; positive high-motion gain; `rho_corr < rho_fixed`; correct Trace beats controls.
- **Conditional GO:** documented visible cameras/regions only; later method uses exactly the same reliability mask and claim scope.
- **NO-GO:** calibration unexplained, semantic correspondence fails, or correct Trace does not beat controls. Do not build the Delta Adapter.

## Immediate next action

Only Step 0.3 is authorized for the next patch: make the raw-camera transform independently testable and export per-query diagnostics/overlay from the existing sanity script. No cache, feature, Motus, or training code changes occur before its review.
