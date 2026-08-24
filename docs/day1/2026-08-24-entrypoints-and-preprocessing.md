# D1 — exact training / RoboTwin evaluation entrypoints

Date: 2026-08-24  
Audit type: read-only source and processor audit. No Motus, Trace, training, or RoboTwin evaluation code changed.

## Scope

Audited workspace: `/kpfs-intern/binz/motus-delta-adapter`, branch `delta-visual-adapter`, base revision `f771216802b8a1601599422f12088bee3c068c14` plus its active uncommitted Delta-adapter files. The actual formal launcher is `scripts/run_delta_ablation.sh`; the older generic and Slurm shells converge on the same Python entrypoint.

## Training call chain

```text
scripts/run_delta_ablation.sh
  └─ launch_training(...)
      └─ .venv/bin/python -m torch.distributed.run train/train.py
          ├─ train.train.main()
          │   ├─ load_config(config YAML)
          │   ├─ create_model_and_optimizer(config)
          │   ├─ create_dataloaders(config)
          │   └─ accelerator.prepare(model, optimizer, train_dataloader, scheduler)
          └─ UniDiffuserTrainer.train()
              └─ UniDiffuserTrainer.train_step(batch)
                  └─ Motus.training_step(...)
                      └─ UndModule.extract_und_features(vlm_inputs, prev_vlm_inputs)
                          ├─ frozen Qwen current / previous forwards
                          ├─ DeltaFeatureAdapter on visual positions
                          └─ VLM adapter → Understanding Expert → MoT
```

There is **no** active `Motus.forward()` method. The actual training model entry is `Motus.training_step()` at `models/motus.py:956` (AST-verified).

```text
create_dataset(config, val)
  └─ RobotWinTaskDataset.__getitem__
      ├─ reads videos/<episode>.mp4
      ├─ selects condition I_t and video/action targets
      ├─ selects previous index max(0, t - temporal_delta_stride)
      ├─ builds VLM input for I_t
      └─ builds VLM input for I_(t-k), or aliases I_t at episode start
          ↓
data.dataset.collate_fn
  ├─ stacks image, video, state and action tensors
  ├─ pads/batches vlm_inputs
  └─ pads/batches prev_vlm_inputs
```

The active `real_previous` config enforces `temporal_delta_stride=3` and `global_downsample_rate=3`. The previous index stays in the same MP4/episode; start-of-episode uses the current VLM input and gives exact zero Delta.

## RoboTwin evaluation call chain

```text
inference/robotwin/Motus/eval.sh             # one task
inference/robotwin/Motus/auto_eval.sh        # task list / GPU scheduler
  └─ $ROBOTWIN_ROOT/script/eval_policy.py
      ├─ loads task config and policy/Motus/deploy_policy.yml
      ├─ imports policy.Motus; calls get_model(usr_args)
      ├─ reset_model(model) for every expert-valid episode
      └─ TASK_ENV.get_obs()
          → policy.Motus.eval(TASK_ENV, model, observation)
          → MotusPolicy.update_obs()
          → MotusPolicy.get_action()
          → Motus.inference_step(...)
          → TASK_ENV.take_action(actions[0], action_type="qpos")
```

`eval_policy.py` prefilters to expert-solvable seeds. Report results as **RoboTwin randomized evaluation** with valid and candidate seed counts, not as an unconditional 100-seed success rate.

The runner contract is documented by [RoboTwin's official deployment guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html) and the [`eval_policy.py` interface](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/script/eval_policy.py). The simulation machine was unreachable during this audit, so its outer runner was cross-checked against the official interface while the Motus policy and training code were inspected directly on the active training host.

### Temporal cache check

The active policy uses `deque(maxlen=2)`. Its first control step supplies the current image as previous image; later steps supply the immediately preceding observation; `reset_model()` clears cache at every episode boundary. This is correct. One remaining runtime check is whether one `TASK_ENV.take_action(actions[0])` corresponds exactly to the dataset's 3 raw-video-frame interval; code order is right, but the simulator clock needs a live host measurement.

## Camera composition verdict

Offline conversion writes a `320×360` composite MP4. A processed training video was measured at `320×360`, 77 frames. Its intended canonical layout is head `[240,320]` above left/right wrists `[120,160]` each. Both paths apply `resize_with_padding(..., (384,320))`, producing the same intended `[3,384,320]` conditioning shape with 12-pixel top/bottom padding.

So the **T-layout placement is structurally compatible**, provided runtime head observation is `[240,320]`; otherwise `np.concatenate` would fail because the wrist row is width 320. It is not bit-identical for 480×640 source cameras: offline conversion makes a half-size wrist then rescales the whole T image, whereas runtime conversion resizes each wrist directly to `160×120`; MP4 compression adds another difference.

## Qwen preprocessing verdict — **not equivalent**

I executed the exact current training and evaluation processor calls with the same deterministic 384×320 RGB PIL image, Qwen3-VL processor, and instruction.

| Quantity | Training | Evaluation | Equal? |
| --- | ---: | ---: | --- |
| `image_grid_thw` | `[1,24,20]` | `[1,24,20]` | yes |
| `pixel_values` shape | `[480,1536]` | `[480,1536]` | shape only |
| `pixel_values` SHA | `c3fc63a720c228a3` | `d0884ed875f613fb` | **no** |
| absolute pixel difference | mean `0.00384927`, max `0.72156858`, nonzero `9.2708%` | — | **no** |
| `input_ids` | `[1,178]` | `[1,175]` | **no** |

Training `utils/vlm_utils.py::preprocess_vlm_messages` uses **image → text**, calls `process_vision_info`, and sets `add_generation_prompt=True`. Evaluation `deploy_policy.py::_preprocess_vlm_messages` uses **text → image**, skips `process_vision_info`, and sets `add_generation_prompt=False`. In the actual test, training transformed the image from `320×384` to `308×392` before processor invocation; evaluation provided `320×384`.

This is a P0 train/test mismatch: the visual grid is both 12×10, but visual tensors and language/image token positions are not equal.

## Required repair (not applied)

Before a closed-loop Delta claim, introduce one shared tested function for T composition, padding, PIL conversion, Qwen chat-template order, `process_vision_info`, and processor invocation. Its parity test must pass the same canonical camera triplet/instruction through the dataset and `MotusPolicy`, asserting equal `pixel_values`, `image_grid_thw`, `input_ids`, and visual placeholder mask. This is a preprocessing-only repair; it does not change Motus, Trace, sampling, or weights.
