#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/tmp/compare_1xh100}"
MODE="${MODE:-quick}"

SUB3_REF="${SUB3_REF:-fork/codex/submission3-fastguard}"
SUB3_PATH_IN_REF="${SUB3_PATH_IN_REF:-submission3/train_gpt.py}"
TOP_REF="${TOP_REF:-origin/main}"
TOP_PATH_IN_REF="${TOP_PATH_IN_REF:-records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py}"

DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_1024_bpe.model}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
SEED="${SEED:-42}"

case "$MODE" in
  quick)
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-180}"
    EVAL_STRIDE="${EVAL_STRIDE:-0}"
    ;;
  confirm)
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-300}"
    EVAL_STRIDE="${EVAL_STRIDE:-64}"
    ;;
  *)
    echo "Unsupported MODE=$MODE (use quick or confirm)" >&2
    exit 1
    ;;
esac

mkdir -p "$WORK_DIR"
SUB3_SCRIPT="$WORK_DIR/submission3_train_gpt.py"
TOP_SCRIPT="$WORK_DIR/top_train_gpt.py"
SUB3_LOG="$WORK_DIR/submission3_${MODE}.log"
TOP_LOG="$WORK_DIR/top_${MODE}.log"

extract_script() {
  local ref="$1"
  local path_in_ref="$2"
  local out="$3"
  git -C "$ROOT_DIR" show "${ref}:${path_in_ref}" > "$out"
}

run_candidate() {
  local name="$1"
  local script="$2"
  local log="$3"

  echo "=== running ${name} (${MODE}) ==="
  (
    cd "$ROOT_DIR"
    RUN_ID="${name}_${MODE}_seed${SEED}" \
    SEED="$SEED" \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    VOCAB_SIZE="$VOCAB_SIZE" \
    MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
    VAL_LOSS_EVERY=0 \
    TRAIN_LOG_EVERY=100 \
    SWA_ENABLED=0 \
    LAWA_ENABLED=0 \
    TTT_ENABLED=0 \
    EVAL_STRIDE="$EVAL_STRIDE" \
    torchrun --standalone --nproc_per_node=1 "$script"
  ) | tee "$log"
}

extract_summary() {
  local name="$1"
  local log="$2"
  python3 - "$name" "$log" <<'PY'
import re
import sys
from pathlib import Path

name = sys.argv[1]
log_path = Path(sys.argv[2])
text = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

train_line = None
for line in text:
    if " train_loss:" in line and "step:" in line:
        train_line = line

exact_line = None
for line in reversed(text):
    if "_exact val_loss:" in line and "val_bpb:" in line:
        exact_line = line
        break

def grab(pattern, line):
    if line is None:
        return "na"
    m = re.search(pattern, line)
    return m.group(1) if m else "na"

step = grab(r"step:(\d+/\d+)", train_line)
step_avg = grab(r"step_avg:([0-9.]+)ms", train_line)
train_loss = grab(r"train_loss:([0-9.]+)", train_line)
val_bpb = grab(r"val_bpb:([0-9.]+)", exact_line)
metric_name = "na"
if exact_line is not None:
    metric_name = exact_line.split()[0]

print(
    f"{name} "
    f"last_train_step={step} "
    f"last_train_loss={train_loss} "
    f"last_step_avg_ms={step_avg} "
    f"final_metric={metric_name} "
    f"final_bpb={val_bpb}"
)
PY
}

print_protocol() {
  cat <<EOF
Protocol:
- mode=$MODE
- 1xH100 via torchrun --nproc_per_node=1
- shared seed=$SEED
- shared max_wallclock_seconds=$MAX_WALLCLOCK_SECONDS
- shared cheap train-time eval: VAL_LOSS_EVERY=0
- expensive extras disabled: SWA_ENABLED=0 LAWA_ENABLED=0 TTT_ENABLED=0
- final eval stride=$EVAL_STRIDE
- submission3 ref: $SUB3_REF:$SUB3_PATH_IN_REF
- top ref: $TOP_REF:$TOP_PATH_IN_REF
EOF
}

print_protocol
extract_script "$SUB3_REF" "$SUB3_PATH_IN_REF" "$SUB3_SCRIPT"
extract_script "$TOP_REF" "$TOP_PATH_IN_REF" "$TOP_SCRIPT"

run_candidate "submission3" "$SUB3_SCRIPT" "$SUB3_LOG"
run_candidate "top" "$TOP_SCRIPT" "$TOP_LOG"

echo "=== summary ==="
extract_summary "submission3" "$SUB3_LOG"
extract_summary "top" "$TOP_LOG"

cat <<'EOF'

How to read this:
- `quick` is the decision run. It compares core training quality under the same 1xH100 wallclock without paying for repeated full eval or legal TTT.
- If the top model only wins by a tiny margin, prefer optimizing submission3 first because it is simpler.
- If the top model wins clearly, port features one at a time into submission3:
  1. activation
  2. smear
  3. bigram
  4. larger MLP
  5. anything more exotic after that

Suggested next step:
- Run MODE=quick first.
- Only run MODE=confirm if the quick result is close enough that you need a higher-confidence call.
EOF
