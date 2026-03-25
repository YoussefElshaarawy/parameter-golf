# Relu3 Clean Residual Fast (No Body)

Status: fast clean-brain submission scaffold with a hard 16 MB export guard.

## Architecture

This variant keeps the clean residual block that tested best locally:

- `relu3`
- no encoder-decoder skip path
- no learned `attn_scale`
- no learned `mlp_scale`
- keep `resid_mix`
- keep `q_gain`
- tied embeddings, GQA, RoPE, RMSNorm

Unlike `submission2`, this version removes the extra body features by default:

- `USE_BIGRAM=0`
- `USE_SMEAR=0`

The goal is a cleaner and lighter large-scale run that stays closer to the `brain_base_fast` direction.

## Methodology

`MiniBase` was the small proxy for the real model. It let me test one architectural change at a time before touching the full training scaffold. The main result was that the cleaner block consistently beat the more routed baseline:

- removing learned skip reuse helped a lot
- removing learned residual scales did not hurt
- `resid_mix` helped a little
- `q_gain` helped a little
- sharper activations helped, and `relu3` remained viable after quantization probing

This submission keeps the strong public training and compression scaffold, but swaps in the cleaner local architecture and a reduced-work fast loop.

## Fast Work Cut

This script cuts per-step work by fixed fractions instead of machine-specific constants:

- `FAST_SEQ_FRAC=0.5`
- `FAST_BATCH_FRAC=0.25`
- `FAST_VAL_BATCH_FRAC=0.25`
- `FAST_EVAL_BATCH_FRAC=0.25`

That keeps the same relative work reduction on laptop or H100.

## Size Guard

`submission3` enforces a hard size cap during export:

- `MAX_SUBMISSION_BYTES=16000000`

If the compressed artifact plus code would exceed that limit, the run fails instead of silently producing an invalid submission.

## Run

```bash
bash prepare.sh
bash eval/eval.sh

SEED=42 bash eval/eval.sh
SEED=1337 bash eval/eval.sh
SEED=2024 bash eval/eval.sh

# Preflight only: print structure and rough size estimates, then exit
PRINT_MODEL_INFO_ONLY=1 bash eval/eval.sh
```

If you want to cut work even harder:

```bash
FAST_SEQ_FRAC=0.5 FAST_BATCH_FRAC=0.125 FAST_VAL_BATCH_FRAC=0.125 bash eval/eval.sh
```

## Notes

This file is intentionally shorter and cleaner in intent than the heavier feature-stacked alternatives. The main objective here is:

- fast large-scale iteration
- clean brain-only architecture
- strict submission validity under the 16 MB limit
