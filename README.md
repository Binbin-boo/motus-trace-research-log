# Motus + TraceAnything research log

Daily, reproducible records for the Motus temporal-delta study. This repository
contains code-change manifests, test commands, numerical outputs, and conclusions;
it deliberately does **not** include RoboTwin frames, Trace caches, model weights,
or the full upstream Motus / TraceAnything source trees.

## Current scope

The first stage tests a narrow hypothesis: correspondence-aligned temporal visual
delta should remove false changes caused by image-coordinate mismatch before any
Motus policy training is attempted.

## Day index

| Day | Date | Question | Record |
| --- | --- | --- | --- |
| D1 | 2026-08-24 | Is the raw temporal-delta problem real, and is Trace coordinate transport auditable? | [D1 record](docs/day1/2026-08-24.md) |

## Source workspace

The reviewed Day-1 implementation lives in the Motus delta-adapter worktree.
The exact files and the tests needed to reproduce it are listed in the D1 record.
Before a formal cache run, regenerate caches under the schema noted there; do not
mix schema-1 and schema-2 records.
