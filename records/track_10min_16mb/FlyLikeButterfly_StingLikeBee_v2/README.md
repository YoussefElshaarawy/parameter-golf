# FlyLikeButterfly_StingLikeBee_v2

Status: record-style submission folder with rare, gentler deep training phases and a hard 16 MB export guard.

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

This record keeps the same fast clean-brain scaffold, but makes the bee phase much rarer and less disruptive:

- fly like a butterfly: train most steps with the fast reduced-work setup
- sting like a bee: every 5000 steps, temporarily switch to a moderately deeper workload to push past shallow plateaus

So this is not a new architecture. It is a new workload schedule for the same clean brain.

## Butterfly / Bee Schedule

Fast butterfly defaults:

- `FAST_SEQ_FRAC=0.5`
- `FAST_BATCH_FRAC=0.25`
- `FAST_VAL_BATCH_FRAC=0.25`
- `FAST_EVAL_BATCH_FRAC=0.25`

Bee sting defaults:

- `STING_INTERVAL=5000`
- `STING_STEPS=32`
- `STING_SEQ_FRAC=0.75`
- `STING_BATCH_FRAC=0.5`

That means the run stays in butterfly mode almost all the time, and only occasionally enters a shorter, deeper bee phase. The bee phase is intentionally stronger than butterfly, but not a full-work shock.

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
