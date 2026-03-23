# FlyLikeButterfly_StingLikeBee

Status: record-style submission folder with periodic deep training phases and a hard 16 MB export guard.

## Architecture

This variant keeps the clean brain that tested best locally:

- `relu3`
- no encoder-decoder skip path
- no learned `attn_scale`
- no learned `mlp_scale`
- keep `resid_mix`
- keep `q_gain`
- tied embeddings, GQA, RoPE, RMSNorm

Body features stay off by default:

- `USE_BIGRAM=0`
- `USE_SMEAR=0`

## Methodology

`MiniBase` was the small proxy for the real model. It let me isolate one change at a time before moving it into the full scaffold. The main result was that the clean residual block outperformed the more routed baseline, and the fast cut regime dramatically improved step speed.

This record adds one new training idea on top of the simpler fast clean-brain scaffold:

- fly like a butterfly: train most steps with the fast reduced-work setup
- sting like a bee: every fixed interval, temporarily switch back to deeper training work to escape shallow plateaus

So this is not a new architecture. It is a new workload schedule for the same clean brain.

## Butterfly / Bee Schedule

Fast mode defaults:

- `FAST_SEQ_FRAC=0.5`
- `FAST_BATCH_FRAC=0.25`
- `FAST_VAL_BATCH_FRAC=0.25`
- `FAST_EVAL_BATCH_FRAC=0.25`

Deep sting defaults:

- `STING_INTERVAL=700`
- `STING_STEPS=64`
- `STING_SEQ_FRAC=1.0`
- `STING_BATCH_FRAC=1.0`

That means the run spends most of its time in fast shallow updates, then periodically switches to full-depth work for a short burst.

## Size Guard

This record enforces:

- `MAX_SUBMISSION_BYTES=16000000`

If the compressed artifact plus code exceeds the limit, the run fails instead of silently producing an invalid submission.

The exported artifact is named:

- `final_model.mixed.ptz`

and the logs report it consistently as:

- `mixed_int6_int8+zstd`

## Run

```bash
bash prepare.sh
bash eval/eval.sh

SEED=42 bash eval/eval.sh
SEED=1337 bash eval/eval.sh
SEED=2024 bash eval/eval.sh

# Preflight only
PRINT_MODEL_INFO_ONLY=1 bash eval/eval.sh
```

## Goal

The purpose of `submission4` is simple:

- keep the clean brain
- keep the fast step loop
- add periodic deeper training bursts so the run does not plateau as early as `submission3`
