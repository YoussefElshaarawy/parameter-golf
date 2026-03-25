#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-submissionSwan_seed${SEED}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

cd "${ROOT_DIR}"

RUN_ID="${RUN_ID}" \
SEED="${SEED}" \
DATA_PATH="${DATA_PATH:-../data/datasets/fineweb10B_sp1024}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-../data/tokenizers/fineweb_1024_bpe.model}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
FAST_SEQ_FRAC="${FAST_SEQ_FRAC:-0.5}" \
FAST_BATCH_FRAC="${FAST_BATCH_FRAC:-0.25}" \
FAST_VAL_BATCH_FRAC="${FAST_VAL_BATCH_FRAC:-0.25}" \
FAST_EVAL_BATCH_FRAC="${FAST_EVAL_BATCH_FRAC:-0.25}" \
MAX_SUBMISSION_BYTES="${MAX_SUBMISSION_BYTES:-16000000}" \
PRINT_MODEL_INFO_ONLY="${PRINT_MODEL_INFO_ONLY:-0}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py | tee "train_seed${SEED}.log"
