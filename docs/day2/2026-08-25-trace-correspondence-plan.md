# Day 2 — Trace correspondence audit plan

Status: **planned; no Day-2 Trace model run or Motus code change has been made.**

## Single research question

For a RoboTwin image from one physical camera, can Trace Anything reliably map a
physical region in the current frame to the same physical region in frame `t-k`
(`k=3` for this audit)?

This is deliberately not a Motus-feature, policy-training, or optical-flow
benchmark.  The only possible conclusion today is **GO**, **Conditional GO**, or
**NO-GO** for using Trace as a correspondence operator in the next stage.

## Non-negotiable scope

- Use `HEAD`, `LEFT_WRIST`, and `RIGHT_WRIST` independently.  Never run Trace
  directly on the three-camera Motus composite.
- Do not modify Motus, Qwen, WAN, Action Expert, the existing inference API, or
  any training configuration.
- Do not alter Trace Anything weights or its algorithm.
- `tools/trace_pair.py` must not import Motus, Qwen, WAN, or Action Expert.
- A nearest neighbour always existing is **not** visibility or correctness.
  Therefore the result will expose geometry, ambiguity, and cycle diagnostics
  separately instead of calling all finite matches `valid`.
- We will inspect the installed Trace Anything version before assuming its exact
  tensor names, trajectory order, confidence parameterisation, or motion-mask
  implementation.
- Every implementation and execution phase stops for user review before the
  next phase begins.

## Coordinate and output contract

The intended primary mapping is backward correspondence in the original pixel
coordinates of one physical camera:

`C_cur_to_prev(x_cur) = x_prev`.

The planned result object will retain raw quantities rather than hide them in a
single score:

```text
backward_xy                    # current -> previous, original camera pixels
forward_xy                     # previous -> current, for the cycle check
src_track_conf / target_track_conf
nn_distance / nn_ratio         # d1 and d1/(d2+eps)
cycle_error
finite_mask
reliable_mask                  # transparent rule, initially finite & cycle only
motion_proxy                   # never named an occlusion mask
occlusion_or_mismatch_candidate
camera, episode, prev_idx, cur_idx, k, transform metadata
```

The 2-D correspondence is a derived operation: Trace trajectories/3-D points
are matched with nearest-neighbour search at the target timestamp.  It will not
be described as a native optical-flow output.

## Review-gated execution sequence

### D2.0 — Source and API contract audit (first; read-only)

No file is changed and no GPU model is loaded.

1. Locate the actual raw RoboTwin per-camera source used by the processed
   Motus samples.  The processed composite MP4 alone is insufficient for this
   audit; cropping it would violate the same-camera requirement.
2. Record actual `HEAD`, `LEFT_WRIST`, and `RIGHT_WRIST` sizes, frame indexing,
   episode identifiers, camera naming, and relation to a processed sample.
3. Inspect the exact installed Trace Anything checkout: preprocessing,
   `forward` signature, prediction dictionary fields/shapes, source-frame
   ordering, timestamp handling, confidence transformation, and any `fg_mask`.
4. Establish the exact forward and inverse camera-to-Trace coordinate
   transforms.  Do not assume a transpose/crop is absent merely because a
   nominal camera aspect ratio is landscape.

Deliverable for review: one short API/data-contract table with exact source
paths and shapes.  No code patch or test result is claimed at this stage.

### D2.1 — Isolated interface and geometry tests (patch review before running)

Proposed new files only:

```text
tools/trace_pair.py
tests/test_trace_pair.py
```

The first patch will contain pure, dependency-light helpers for:

```text
prepare_camera_pair
build_backward_correspondence
build_reverse_correspondence
compute_reliability
visualize_pair                 # declaration/interface only if useful
```

Tests will use synthetic trajectories/coordinates rather than a GPU model:

1. identity mapping: `C(x) ~= x`;
2. known horizontal translation: current-to-previous sign and magnitude are
   correct;
3. reverse mapping produces near-zero cycle error for a consistent field;
4. the deliberately wrong direction fails the known-direction assertion;
5. same-camera metadata is mandatory and coordinate outputs are finite.

Deliverable for review: exact diff first, then test command and output after
approval.  This phase has no Motus/Qwen imports.

### D2.2 — Trace extraction and basic sanity checks (run review before running)

After the interface is approved, call the installed frozen Trace model directly
on one real per-camera pair.  Do not rely on a compact official `output.pt` if
it drops trajectory tensors needed for matching.

Run, in order:

1. real-image identity pair `I_t, I_t`;
2. synthetic pixel translation of one real camera image with known backward
   displacement;
3. one real `I_(t-3), I_t` pair per camera.

Report only measured quantities: input/Trace sizes, median and 95th percentile
identity displacement, translation endpoint error, finite fraction, timing,
and peak GPU memory.  A failed identity or sign test is a stop condition.

### D2.3 — Derived correspondence and reliability diagnostics

Extend the reviewed code to obtain backward and reverse mappings from the
actual Trace outputs.  Compute `d1`, `d2`, `d1/(d2+eps)`, and cycle error in
the same original per-camera coordinate system.

Initial reliability is intentionally simple:

```text
finite_mask AND (cycle_error < tau_cycle)
```

Trace confidence and NN ambiguity are saved for analysis, not tuned into a
hidden score.  Failure of a cycle check is reported as an
`occlusion_or_mismatch_candidate`, never as ground-truth occlusion.

Deliverable for review: one real pair per camera with coloured point-to-point
matches, displacement field, confidence, NN diagnostics, cycle error,
reliability mask, and motion proxy.

### D2.4 — Sampling and manual-review package

Only after the one-pair visualizations look semantically plausible:

1. add a manifest generator for 100 pairs (`k=3`), aiming for 60 targeted and
   40 random cases across all three cameras;
2. preserve episode/task/frame/camera metadata so no cross-episode or
   cross-camera pair can enter;
3. save review pages with point-to-point correspondences as the primary view,
   not only colour-flow images;
4. keep labels separate from model predictions.

The targeted manifest will mark candidate situations (static, arm, manipulated
object, wrist-camera motion, occlusion/contact/disocclusion).  It does not
pretend these are ground truth until manual review.

Manual labels for each inspected correspondence:

```text
2  clearly the same physical point/part
1  plausible but ambiguous
0  clearly wrong object/part/background
X  no true correspondence (occluded or newly visible)
```

Before the full 100-pair run, the manifest, query-point density, visual layout,
output paths, and run command are separately reviewed.

### D2.5 — Analysis and decision

Report counts by category:

| category | N | clearly correct | ambiguous | wrong | no correspondence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static background | | | | | |
| Robot arm | | | | | |
| Manipulated object | | | | | |
| Camera motion | | | | | |
| Occlusion/disocclusion | | | | | |

Also compare, for manually correct versus clearly wrong points:

| diagnostic | correct median | wrong median |
| --- | ---: | ---: |
| Trace track confidence | | |
| NN distance | | |
| NN ratio | | |
| cycle error | | |

Decision rule:

- **GO:** visible static/robot/object matching is reliable, wrist-camera
  motion remains usable, and cycle error has useful failure-filtering value.
- **Conditional GO:** visible matching is useful but occlusion/confidence is
  weak; future aligned delta must mask only transparent reliable matches.
- **NO-GO:** manipulated objects frequently jump to robot/background, camera
  motion causes broad collapse, or no diagnostic separates obvious failures.

## Expected code/change inventory

| Phase | File | Action |
| --- | --- | --- |
| D2.0 | `docs/day2/...` | Add this plan and later factual audit record only |
| D2.1 | `tools/trace_pair.py` | New isolated correspondence utility |
| D2.1 | `tests/test_trace_pair.py` | New synthetic coordinate tests |
| D2.4 | `tools/sample_trace_pairs.py` | New, only if sampling is approved |
| D2.4 | `tools/review_trace_pairs.py` | New, only if visualization is approved |
| all | `docs/day2/...` | Record commands, results, and GO decision |

No generated images, model outputs, raw data, checkpoints, or credentials are
committed to the research-log repository.

## Execution handoff

This plan is intentionally executed inline with the user rather than as a
single autonomous batch.  The next permitted action is **D2.0 read-only audit**;
we stop and present its factual table before proposing the D2.1 code patch.
