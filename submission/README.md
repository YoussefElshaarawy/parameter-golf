# Relu3 Clean Residual + BigramHash + SmearGate

Status: first submission scaffold. Update the metrics below after the 8xH100 runs complete.

## Goal

This folder is structured like a Parameter Golf record submission:

- `train_gpt.py`
- `submission.json`
- `README.md`
- per-seed logs such as `train_seed42.log`

The script is built on a competitive public scaffold, but changes the core block methodology:

- default activation is `relu3`
- no encoder→decoder skip path
- no learned `attn_scale`
- no learned `mlp_scale`
- keep `resid_mix`
- keep `q_gain`
- keep tied embeddings, GQA, RoPE, RMSNorm

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

## Run Command

```bash
# Setup once
bash prepare.sh

# Train + evaluate on 8xH100
bash eval/eval.sh

# Specific seed
SEED=42 bash eval/eval.sh
SEED=1337 bash eval/eval.sh
SEED=2024 bash eval/eval.sh
```

All important defaults are already encoded in `train_gpt.py`.

## What To Copy Into The Submission

After each run, the script emits the key lines needed for the writeup:

- `Serialized model int6+zstd: ... bytes`
- `Total submission size int6+zstd: ... bytes`
- `final_submission_roundtrip_exact val_loss:... val_bpb:...`
- `submission_summary: ...`

Use the exact `final_submission_roundtrip_exact` line for the score in the README and `submission.json`.

## Local Evidence For This Methodology

The local MiniBase quantization probe in this workspace showed:

- `relu3` trained to lower validation loss than `relu2`
- `relu3` had a larger quantization penalty than `relu2`
- `relu3` still remained better after quantization in the probe

That is why this submission keeps the stronger challenge scaffold and swaps in the cleaner `relu3` residual block, instead of submitting the small proxy architecture directly.

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
