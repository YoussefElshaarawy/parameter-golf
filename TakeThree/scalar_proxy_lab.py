"""
Human-scale scalar proxy-eval lab.

This is intentionally tiny and readable:
- pure Python scalar autograd values
- char-level dataset from TakeThree/input.txt
- deterministic micro / mid / full eval banks
- compact machine-readable metrics

Use this to sanity-check whether proxy trends track the final full metric
before moving the same idea into a larger tensor harness.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path


def data_file() -> Path:
    return Path(__file__).resolve().parent / "input.txt"


class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def exp(self):
        out = math.exp(self.data)
        return Value(out, (self,), (out,))

    def log(self):
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def relu(self):
        return Value(max(0.0, self.data), (self,), (1.0 if self.data > 0 else 0.0,))

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)


@dataclass(frozen=True)
class EvalWindow:
    doc_idx: int
    start_offset: int
    seq_len: int


@dataclass
class Config:
    seed: int = int(os.environ.get("SEED", "42"))
    train_steps: int = int(os.environ.get("TRAIN_STEPS", "300"))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", "25"))
    eval_every: int = int(os.environ.get("EVAL_EVERY", "0"))

    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", "6"))
    mid_seq_len: int = int(os.environ.get("MID_SEQ_LEN", "10"))
    full_seq_len: int = int(os.environ.get("FULL_SEQ_LEN", "12"))

    proxy_micro_num_windows: int = int(os.environ.get("PROXY_MICRO_NUM_WINDOWS", "8"))
    proxy_mid_num_windows: int = int(os.environ.get("PROXY_MID_NUM_WINDOWS", "24"))
    proxy_micro_every: int = int(os.environ.get("PROXY_MICRO_EVERY", "25"))
    proxy_mid_every: int = int(os.environ.get("PROXY_MID_EVERY", "100"))
    proxy_seed: int = int(os.environ.get("PROXY_SEED", "12345"))

    n_embd: int = int(os.environ.get("MODEL_DIM", "6"))
    hidden_mult: int = int(os.environ.get("HIDDEN_MULT", "2"))
    learning_rate: float = float(os.environ.get("LR", "0.03"))


def load_docs() -> list[str]:
    docs = [line.strip() for line in data_file().read_text(encoding="utf-8").splitlines() if line.strip()]
    if not docs:
        raise ValueError("input.txt is empty")
    return docs


def softmax(logits: list[Value]) -> list[Value]:
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    denom = sum(exps)
    return [e / denom for e in exps]


def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    return [sum(wij * xj for wij, xj in zip(row, x)) for row in w]


def zero_vec(n: int) -> list[Value]:
    return [Value(0.0) for _ in range(n)]


class ScalarLM:
    def __init__(self, vocab_size: int, max_seq_len: int, cfg: Config):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_embd = cfg.n_embd
        hidden = cfg.n_embd * cfg.hidden_mult
        scale = 0.08

        def matrix(rows: int, cols: int) -> list[list[Value]]:
            return [[Value(random.gauss(0.0, scale)) for _ in range(cols)] for _ in range(rows)]

        self.tok_emb = matrix(vocab_size, cfg.n_embd)
        self.pos_emb = matrix(max_seq_len, cfg.n_embd)
        self.fc1 = matrix(hidden, cfg.n_embd)
        self.fc2 = matrix(cfg.n_embd, hidden)
        self.lm_head = matrix(vocab_size, cfg.n_embd)

        self.params = [
            p
            for mat in (self.tok_emb, self.pos_emb, self.fc1, self.fc2, self.lm_head)
            for row in mat
            for p in row
        ]

    def hidden(self, token_id: int, pos_id: int) -> list[Value]:
        x = [a + b for a, b in zip(self.tok_emb[token_id], self.pos_emb[pos_id])]
        h = [v.relu() for v in linear(x, self.fc1)]
        return [a + b for a, b in zip(x, linear(h, self.fc2))]

    def logits(self, token_id: int, pos_id: int) -> list[Value]:
        return linear(self.hidden(token_id, pos_id), self.lm_head)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0

    def step(self, lr: float) -> None:
        for p in self.params:
            p.data -= lr * p.grad


def build_vocab(docs: list[str]) -> tuple[list[str], dict[str, int], int]:
    chars = sorted(set("".join(docs)))
    bos = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    return chars, stoi, bos


def encode_docs(docs: list[str], stoi: dict[str, int], bos: int) -> list[list[int]]:
    encoded = []
    for doc in docs:
        encoded.append([bos] + [stoi[ch] for ch in doc] + [bos])
    return encoded


def token_byte_sizes(chars: list[str], bos: int) -> list[int]:
    sizes = [len(ch.encode("utf-8")) for ch in chars]
    sizes.append(1 if bos >= 0 else 0)
    return sizes


def build_eval_bank(
    sequences: list[list[int]],
    num_windows: int,
    seq_len: int,
    seed: int,
) -> list[EvalWindow]:
    counts = [max(0, len(seq) - seq_len) for seq in sequences]
    total = sum(counts)
    if total <= 0:
        raise ValueError(f"No validation windows for seq_len={seq_len}")
    num_windows = min(num_windows, total)
    rng = random.Random(seed)
    sampled = sorted(rng.sample(range(total), num_windows))
    bank = []
    doc_idx = 0
    base = 0
    for idx in sampled:
        while idx >= base + counts[doc_idx]:
            base += counts[doc_idx]
            doc_idx += 1
        bank.append(EvalWindow(doc_idx=doc_idx, start_offset=idx - base, seq_len=seq_len))
    return bank


def build_full_eval_bank(sequences: list[list[int]], seq_len: int) -> list[EvalWindow]:
    bank = []
    for doc_idx, seq in enumerate(sequences):
        for start in range(0, max(0, len(seq) - seq_len)):
            bank.append(EvalWindow(doc_idx=doc_idx, start_offset=start, seq_len=seq_len))
    if not bank:
        raise ValueError(f"No full eval windows for seq_len={seq_len}")
    return bank


def window_loss_and_bytes(
    model: ScalarLM,
    sequence: list[int],
    start: int,
    seq_len: int,
    token_bytes: list[int],
) -> tuple[Value, int]:
    losses = []
    byte_count = 0
    for pos in range(seq_len):
        token_id = sequence[start + pos]
        target_id = sequence[start + pos + 1]
        probs = softmax(model.logits(token_id, pos))
        losses.append(-probs[target_id].log())
        byte_count += token_bytes[target_id]
    return (sum(losses) / seq_len), byte_count


def run_eval_bank(
    model: ScalarLM,
    sequences: list[list[int]],
    bank: list[EvalWindow],
    token_bytes: list[int],
) -> tuple[float, float]:
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0
    for window in bank:
        loss, byte_count = window_loss_and_bytes(
            model,
            sequences[window.doc_idx],
            window.start_offset,
            window.seq_len,
            token_bytes,
        )
        total_loss += loss.data * window.seq_len
        total_tokens += window.seq_len
        total_bytes += byte_count
    avg_loss = total_loss / total_tokens
    bpb = (avg_loss / math.log(2.0)) * (total_tokens / max(total_bytes, 1))
    return avg_loss, bpb


def format_metric(value: float | None) -> str:
    return "na" if value is None else f"{value:.6f}"


def main() -> None:
    cfg = Config()
    random.seed(cfg.seed)

    docs = load_docs()
    random.shuffle(docs)
    split = max(1, int(0.9 * len(docs)))
    train_docs = docs[:split]
    val_docs = docs[split:] if split < len(docs) else docs[-max(1, len(docs) // 10) :]

    chars, stoi, bos = build_vocab(docs)
    vocab_size = len(chars) + 1
    token_bytes = token_byte_sizes(chars, bos)
    train_sequences = encode_docs(train_docs, stoi, bos)
    val_sequences = encode_docs(val_docs, stoi, bos)

    max_seq_len = max(cfg.train_seq_len, cfg.mid_seq_len, cfg.full_seq_len)
    model = ScalarLM(vocab_size=vocab_size, max_seq_len=max_seq_len, cfg=cfg)

    micro_bank = build_eval_bank(val_sequences, cfg.proxy_micro_num_windows, cfg.train_seq_len, cfg.proxy_seed)
    mid_bank = build_eval_bank(val_sequences, cfg.proxy_mid_num_windows, cfg.mid_seq_len, cfg.proxy_seed + 1)
    full_bank = build_full_eval_bank(val_sequences, cfg.full_seq_len)

    print(f"model_params={len(model.params)}")
    print(
        f"data train_docs={len(train_docs)} val_docs={len(val_docs)} "
        f"train_seq_len={cfg.train_seq_len} mid_seq_len={cfg.mid_seq_len} full_seq_len={cfg.full_seq_len}"
    )
    print(
        "proxy_config "
        f"proxy_micro_num_windows={cfg.proxy_micro_num_windows} "
        f"proxy_mid_num_windows={cfg.proxy_mid_num_windows} "
        f"proxy_micro_every={cfg.proxy_micro_every} "
        f"proxy_mid_every={cfg.proxy_mid_every} "
        f"proxy_seed={cfg.proxy_seed} "
        f"eval_every={cfg.eval_every}"
    )
    print(
        "proxy_bank_meta "
        f"micro_bank_size={len(micro_bank)} "
        f"mid_bank_size={len(mid_bank)} "
        f"full_bank_size={len(full_bank)}"
    )

    last_micro_bpb = None
    last_mid_bpb = None
    last_full_bpb = None
    best_micro_bpb = None
    best_mid_bpb = None
    eval_rows = []

    t0 = time.perf_counter()
    for step in range(1, cfg.train_steps + 1):
        seq = random.choice(train_sequences)
        if len(seq) < cfg.train_seq_len + 1:
            continue
        start = random.randint(0, len(seq) - cfg.train_seq_len - 1)
        loss, _ = window_loss_and_bytes(model, seq, start, cfg.train_seq_len, token_bytes)

        model.zero_grad()
        loss.backward()
        model.step(cfg.learning_rate)

        ran_eval = False
        if cfg.proxy_micro_every > 0 and step % cfg.proxy_micro_every == 0:
            _, last_micro_bpb = run_eval_bank(model, val_sequences, micro_bank, token_bytes)
            best_micro_bpb = last_micro_bpb if best_micro_bpb is None else min(best_micro_bpb, last_micro_bpb)
            ran_eval = True
        if cfg.proxy_mid_every > 0 and step % cfg.proxy_mid_every == 0:
            _, last_mid_bpb = run_eval_bank(model, val_sequences, mid_bank, token_bytes)
            best_mid_bpb = last_mid_bpb if best_mid_bpb is None else min(best_mid_bpb, last_mid_bpb)
            ran_eval = True
        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            _, last_full_bpb = run_eval_bank(model, val_sequences, full_bank, token_bytes)
            ran_eval = True

        avg_step_ms = 1000.0 * (time.perf_counter() - t0) / step
        if step <= 10 or step % cfg.train_log_every == 0 or ran_eval or step == cfg.train_steps:
            print(
                "metrics "
                f"step={step} "
                f"train_loss={loss.data:.6f} "
                f"avg_step_ms={avg_step_ms:.3f} "
                f"proxy_bpb_micro={format_metric(last_micro_bpb)} "
                f"proxy_bpb_mid={format_metric(last_mid_bpb)} "
                f"full_bpb={format_metric(last_full_bpb)} "
                f"best_proxy_bpb_micro={format_metric(best_micro_bpb)} "
                f"best_proxy_bpb_mid={format_metric(best_mid_bpb)}"
            )
        if ran_eval:
            eval_rows.append(
                {
                    "step": step,
                    "train_loss": round(loss.data, 8),
                    "avg_step_ms": round(avg_step_ms, 4),
                    "proxy_bpb_micro": None if last_micro_bpb is None else round(last_micro_bpb, 8),
                    "proxy_bpb_mid": None if last_mid_bpb is None else round(last_mid_bpb, 8),
                    "full_bpb": None if last_full_bpb is None else round(last_full_bpb, 8),
                    "best_proxy_bpb_micro": None if best_micro_bpb is None else round(best_micro_bpb, 8),
                    "best_proxy_bpb_mid": None if best_mid_bpb is None else round(best_mid_bpb, 8),
                }
            )

    if last_full_bpb is None:
        _, last_full_bpb = run_eval_bank(model, val_sequences, full_bank, token_bytes)

    print("final_summary_begin")
    print(f"last_proxy_bpb_micro={format_metric(last_micro_bpb)}")
    print(f"best_proxy_bpb_micro={format_metric(best_micro_bpb)}")
    print(f"last_proxy_bpb_mid={format_metric(last_mid_bpb)}")
    print(f"best_proxy_bpb_mid={format_metric(best_mid_bpb)}")
    print(f"final_full_bpb={format_metric(last_full_bpb)}")
    print(f"correlation_rows_json={json.dumps(eval_rows, separators=(',', ':'))}")
    print("final_summary_end")


if __name__ == "__main__":
    main()
