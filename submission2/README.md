# Relu3 Clean Residual + BigramHash + SmearGate (Fast Percentage Cut)

Status: fast submission scaffold with percentage-based workload cuts. Update the metrics below after the 8xH100 runs complete.

## Goal

This folder is structured like a Parameter Golf record submission:

- `train_gpt.py`
- `submission.json`
- `README.md`
- per-seed logs such as `train_seed42.log`

The script is built on the same competitive public scaffold, but runs a reduced fraction of per-step work by default:

- `FAST_SEQ_FRAC=0.5`
- `FAST_BATCH_FRAC=0.25`
- `FAST_VAL_BATCH_FRAC=0.25`
- `FAST_EVAL_BATCH_FRAC=0.25`

This preserves the same architecture while making the runtime workload smaller on any machine.


- default activation is `relu3`
- no encoder→decoder skip path
- no learned `attn_scale`
- no learned `mlp_scale`
- I kept the `resid_mix`
- I kept the `q_gain`
- I kept tied embeddings, GQA, RoPE, RMSNorm

It preserves the techniques that matter most for challenge competitiveness:

- 10 layers, width 512, MLP 3x
- BigramHash(10240, dim=128)
- SmearGate
- orthogonal init with scaled output projections
- Muon + AdamW split optimizer
- 3% magnitude pruning
- mixed int5/int6 post-training quantization
- `zstd` compression
- sliding-window validation
- SWA over the converged tail

## Method Lineage

This submission was not designed in isolation. It was influenced by two sources:

- the public 10-layer `Int5-MLP + BigramHash(10240)` record (`val_bpb 1.1428`, 2026-03-20), which established a strong practical scaffold around mixed int5/int6 quantization, BigramHash, SWA, and `WD=0.04`
- my own local ablation work, where I tested cleaner residual-block assumptions, `relu3`, quantization behavior and fast percentage-cut training regimes

So the design here is a hybrid:

- public record-style systems scaffold for compression and competitiveness
- local ablation-driven decisions for the block behavior and training-speed tradeoffs

## Methodology

`MiniBase` was the small proxy for the real model. It kept the important structural parts of the real setup, but made them cheap enough to run quickly on my machine. That let me change one thing at a time, keep everything else fixed, and decide whether a change was actually helping before moving it into the full training scaffold.

The main architectural conclusion from those local ablations was that the cleaner residual block worked better:

- remove encoder-decoder skip reuse
- remove learned `attn_scale`
- remove learned `mlp_scale`
- keep `resid_mix`
- keep `q_gain`
- move from `relu2` to `relu3`


## Run Command

```bash
# Setup once
bash prepare.sh

# Train + evaluate on 8xH100 with the default percentage cuts
bash eval/eval.sh

# Specific seed
SEED=42 bash eval/eval.sh
SEED=1337 bash eval/eval.sh
SEED=2024 bash eval/eval.sh

# Override the work cut if needed
FAST_SEQ_FRAC=0.5 FAST_BATCH_FRAC=0.125 FAST_VAL_BATCH_FRAC=0.125 bash eval/eval.sh
```

All important defaults are encoded in `train_gpt.py` and `eval/eval.sh`.

## What To Copy Into The Submission

After each run, the script emits the key lines needed for the writeup:

- `Serialized model int6+zstd: ... bytes`
- `Total submission size int6+zstd: ... bytes`
- `final_submission_roundtrip_exact val_loss:... val_bpb:...`
- `submission_summary: ...`

Use the exact `final_submission_roundtrip_exact` line for the score in the README and `submission.json`.

## Local Evidence For This Methodology

The local proxy experiments pointed in a consistent direction:

- removing learned skip routing helped a lot
- removing learned residual scales did not hurt
- `q_gain` helped a little
- `resid_mix` helped a little
- sharper activations helped, and `relu3` looked safe enough after quantization probing

That is why this submission keeps the stronger public systems scaffold, but replaces the block behavior with the cleaner local architecture instead of copying the public record unchanged.

## Fields To Update After 8xH100 Runs

Replace the placeholders below after the three real runs finish:

- mean `val_bpb`
- standard deviation
- per-seed artifact bytes
- whether each run is under 16,000,000 bytes total
- short blurb in `submission.json`

## Suggested README Result Block

```text
Seed | val_bpb | artifact_bytes | valid
42   | TBD     | TBD            | TBD
1337 | TBD     | TBD            | TBD
2024 | TBD     | TBD            | TBD
Mean | TBD     |                |
Std  | TBD     |                |
```
