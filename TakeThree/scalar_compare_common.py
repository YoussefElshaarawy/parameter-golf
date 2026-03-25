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


@dataclass(frozen=True)
class VariantSpec:
    name: str
    n_layer: int
    n_embd: int
    n_head: int
    mlp_mult: int
    activation: str
    use_skip_path: bool
    use_attn_scale: bool
    use_mlp_scale: bool
    use_resid_mix: bool
    use_q_gain: bool
    use_smear: bool
    use_bigram: bool


BASE_MODEL_SPEC = VariantSpec(
    name="BaseModelScalar",
    n_layer=2,
    n_embd=8,
    n_head=2,
    mlp_mult=2,
    activation="relu2",
    use_skip_path=True,
    use_attn_scale=True,
    use_mlp_scale=True,
    use_resid_mix=True,
    use_q_gain=True,
    use_smear=False,
    use_bigram=False,
)

SUBMISSION3_SPEC = VariantSpec(
    name="Submission3Scalar",
    n_layer=2,
    n_embd=8,
    n_head=2,
    mlp_mult=3,
    activation="relu3",
    use_skip_path=False,
    use_attn_scale=False,
    use_mlp_scale=False,
    use_resid_mix=True,
    use_q_gain=True,
    use_smear=False,
    use_bigram=False,
)

TOP_LEADERBOARD_SPEC = VariantSpec(
    name="TopLeaderboardScalar",
    n_layer=2,
    n_embd=8,
    n_head=2,
    mlp_mult=3,
    activation="lrelu_square",
    use_skip_path=True,
    use_attn_scale=True,
    use_mlp_scale=True,
    use_resid_mix=True,
    use_q_gain=True,
    use_smear=True,
    use_bigram=True,
)


@dataclass
class TrainConfig:
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

    learning_rate: float = float(os.environ.get("LR", "0.03"))


def softmax(logits: list[Value]) -> list[Value]:
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    denom = sum(exps)
    return [e / denom for e in exps]


def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    return [sum(wij * xj for wij, xj in zip(row, x)) for row in w]


def sigmoid(x: Value) -> Value:
    return Value(1.0) / (Value(1.0) + (-x).exp())


def rmsnorm(x: list[Value], eps: float = 1e-5) -> list[Value]:
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + eps) ** -0.5
    return [xi * scale for xi in x]


def apply_activation(xs: list[Value], kind: str) -> list[Value]:
    if kind == "relu3":
        return [x.relu() ** 3 for x in xs]
    if kind == "lrelu_square":
        return [((x.relu()) + (Value(0.5) * ((-x).relu()) * -1.0)) ** 2 for x in xs]
    return [x.relu() ** 2 for x in xs]


def zeros(n: int) -> list[Value]:
    return [Value(0.0) for _ in range(n)]


class ScalarTransformer:
    def __init__(self, vocab_size: int, max_seq_len: int, spec: VariantSpec):
        if spec.n_embd % spec.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.spec = spec
        self.head_dim = spec.n_embd // spec.n_head
        self.bigram_vocab_size = 64
        scale = 0.08

        def matrix(rows: int, cols: int) -> list[list[Value]]:
            return [[Value(random.gauss(0.0, scale)) for _ in range(cols)] for _ in range(rows)]

        def vector(size: int, fill: float) -> list[Value]:
            return [Value(fill) for _ in range(size)]

        self.tok_emb = matrix(vocab_size, spec.n_embd)
        self.pos_emb = matrix(max_seq_len, spec.n_embd)
        self.q = [matrix(spec.n_embd, spec.n_embd) for _ in range(spec.n_layer)]
        self.k = [matrix(spec.n_embd, spec.n_embd) for _ in range(spec.n_layer)]
        self.v = [matrix(spec.n_embd, spec.n_embd) for _ in range(spec.n_layer)]
        self.o = [matrix(spec.n_embd, spec.n_embd) for _ in range(spec.n_layer)]
        hidden = spec.n_embd * spec.mlp_mult
        self.fc1 = [matrix(hidden, spec.n_embd) for _ in range(spec.n_layer)]
        self.fc2 = [matrix(spec.n_embd, hidden) for _ in range(spec.n_layer)]
        self.attn_scale = [vector(spec.n_embd, 1.0) for _ in range(spec.n_layer)] if spec.use_attn_scale else None
        self.mlp_scale = [vector(spec.n_embd, 1.0) for _ in range(spec.n_layer)] if spec.use_mlp_scale else None
        self.resid_mix = (
            [(vector(spec.n_embd, 1.0), vector(spec.n_embd, 0.0)) for _ in range(spec.n_layer)]
            if spec.use_resid_mix
            else None
        )
        self.q_gain = [vector(spec.n_head, 1.5) for _ in range(spec.n_layer)] if spec.use_q_gain else None
        self.skip_weights = (
            [vector(spec.n_embd, 1.0) for _ in range(spec.n_layer // 2)]
            if spec.use_skip_path and spec.n_layer > 1
            else None
        )
        self.smear_gate = vector(spec.n_embd, 0.0) if spec.use_smear else None
        self.bigram_emb = matrix(self.bigram_vocab_size, spec.n_embd) if spec.use_bigram else None
        self.lm_head = self.tok_emb

        mats = [self.tok_emb, self.pos_emb, self.lm_head]
        mats.extend(self.q + self.k + self.v + self.o + self.fc1 + self.fc2)
        if self.bigram_emb is not None:
            mats.append(self.bigram_emb)
        self.params = [p for mat in mats for row in mat for p in row]
        if self.attn_scale is not None:
            self.params.extend(p for vec in self.attn_scale for p in vec)
        if self.mlp_scale is not None:
            self.params.extend(p for vec in self.mlp_scale for p in vec)
        if self.resid_mix is not None:
            self.params.extend(p for pair in self.resid_mix for vec in pair for p in vec)
        if self.q_gain is not None:
            self.params.extend(p for vec in self.q_gain for p in vec)
        if self.skip_weights is not None:
            self.params.extend(p for vec in self.skip_weights for p in vec)
        if self.smear_gate is not None:
            self.params.extend(self.smear_gate)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0

    def step(self, lr: float) -> None:
        for p in self.params:
            p.data -= lr * p.grad

    def bigram_hash(self, prev_token: int, token: int) -> int:
        return ((prev_token * 131) ^ (token * 17)) % self.bigram_vocab_size

    def apply_smear(self, x: list[Value], prev_x: list[Value] | None) -> list[Value]:
        if self.smear_gate is None or prev_x is None:
            return x
        out = []
        for xi, pi, gi in zip(x, prev_x, self.smear_gate):
            g = sigmoid(gi)
            out.append((Value(1.0) - g) * xi + g * pi)
        return out

    def attention(self, layer: int, x: list[Value], keys: list[list[list[Value]]], values: list[list[list[Value]]]) -> list[Value]:
        q = linear(rmsnorm(x), self.q[layer])
        k = linear(rmsnorm(x), self.k[layer])
        v = linear(rmsnorm(x), self.v[layer])
        keys[layer].append(k)
        values[layer].append(v)
        out = []
        for head in range(self.spec.n_head):
            hs = head * self.head_dim
            qh = q[hs : hs + self.head_dim]
            if self.q_gain is not None:
                qh = [self.q_gain[layer][head] * qv for qv in qh]
            kh = [kk[hs : hs + self.head_dim] for kk in keys[layer]]
            vh = [vv[hs : hs + self.head_dim] for vv in values[layer]]
            scores = []
            for past in kh:
                score = sum(a * b for a, b in zip(qh, past)) / math.sqrt(self.head_dim)
                scores.append(score)
            weights = softmax(scores)
            head_out = []
            for dim_idx in range(self.head_dim):
                head_out.append(sum(weights[t] * vh[t][dim_idx] for t in range(len(vh))))
            out.extend(head_out)
        out = linear(out, self.o[layer])
        if self.attn_scale is not None:
            out = [oi * si for oi, si in zip(out, self.attn_scale[layer])]
        return out

    def mlp(self, layer: int, x: list[Value]) -> list[Value]:
        hidden = apply_activation(linear(rmsnorm(x), self.fc1[layer]), self.spec.activation)
        out = linear(hidden, self.fc2[layer])
        if self.mlp_scale is not None:
            out = [oi * si for oi, si in zip(out, self.mlp_scale[layer])]
        return out

    def logits_for_step(self, token_id: int, pos_id: int, prev_token: int | None, prev_input: list[Value] | None, keys, values) -> tuple[list[Value], list[Value]]:
        x = [a + b for a, b in zip(self.tok_emb[token_id], self.pos_emb[pos_id])]
        if self.bigram_emb is not None and prev_token is not None:
            x = [a + b for a, b in zip(x, self.bigram_emb[self.bigram_hash(prev_token, token_id)])]
        x = self.apply_smear(x, prev_input)
        x0 = x
        skips = []
        split = self.spec.n_layer // 2
        for layer in range(self.spec.n_layer):
            x_in = x
            if self.resid_mix is not None:
                mix_a, mix_b = self.resid_mix[layer]
                x_in = [a * xi + b * x0i for a, b, xi, x0i in zip(mix_a, mix_b, x, x0)]
            attn_out = self.attention(layer, x_in, keys, values)
            x = [xi + ai for xi, ai in zip(x_in, attn_out)]
            mlp_out = self.mlp(layer, x)
            x = [xi + mi for xi, mi in zip(x, mlp_out)]
            if self.skip_weights is not None:
                if layer < split:
                    skips.append(x)
                elif skips:
                    skip = skips.pop()
                    weight = self.skip_weights[min(layer - split, len(self.skip_weights) - 1)]
                    x = [xi + wi * si for xi, wi, si in zip(x, weight, skip)]
        logits = linear(rmsnorm(x), self.lm_head)
        return logits, x0


def load_docs() -> list[str]:
    docs = [line.strip() for line in data_file().read_text(encoding="utf-8").splitlines() if line.strip()]
    if not docs:
        raise ValueError("input.txt is empty")
    return docs


def build_vocab(docs: list[str]) -> tuple[list[str], dict[str, int], int]:
    chars = sorted(set("".join(docs)))
    bos = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    return chars, stoi, bos


def encode_docs(docs: list[str], stoi: dict[str, int], bos: int) -> list[list[int]]:
    return [[bos] + [stoi[ch] for ch in doc] + [bos] for doc in docs]


def token_byte_sizes(chars: list[str], bos: int) -> list[int]:
    sizes = [len(ch.encode("utf-8")) for ch in chars]
    sizes.append(1 if bos >= 0 else 0)
    return sizes


def build_eval_bank(sequences: list[list[int]], num_windows: int, seq_len: int, seed: int) -> list[EvalWindow]:
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


def window_loss_and_bytes(model: ScalarTransformer, sequence: list[int], start: int, seq_len: int, token_bytes: list[int]) -> tuple[Value, int]:
    keys = [[] for _ in range(model.spec.n_layer)]
    values = [[] for _ in range(model.spec.n_layer)]
    losses = []
    byte_count = 0
    prev_input = None
    prev_token = None
    for pos in range(seq_len):
        token_id = sequence[start + pos]
        target_id = sequence[start + pos + 1]
        logits, prev_input = model.logits_for_step(token_id, pos, prev_token, prev_input, keys, values)
        probs = softmax(logits)
        losses.append(-probs[target_id].log())
        byte_count += token_bytes[target_id]
        prev_token = token_id
    return (sum(losses) / seq_len), byte_count


def run_eval_bank(model: ScalarTransformer, sequences: list[list[int]], bank: list[EvalWindow], token_bytes: list[int]) -> tuple[float, float]:
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0
    for window in bank:
        loss, byte_count = window_loss_and_bytes(model, sequences[window.doc_idx], window.start_offset, window.seq_len, token_bytes)
        total_loss += loss.data * window.seq_len
        total_tokens += window.seq_len
        total_bytes += byte_count
    avg_loss = total_loss / total_tokens
    bpb = (avg_loss / math.log(2.0)) * (total_tokens / max(total_bytes, 1))
    return avg_loss, bpb


def format_metric(value: float | None) -> str:
    return "na" if value is None else f"{value:.6f}"


def run_variant(spec: VariantSpec) -> None:
    cfg = TrainConfig()
    random.seed(cfg.seed)

    docs = load_docs()
    random.shuffle(docs)
    split = max(1, int(0.9 * len(docs)))
    train_docs = docs[:split]
    val_docs = docs[split:] if split < len(docs) else docs[-max(1, len(docs) // 10) :]

    chars, stoi, bos = build_vocab(docs)
    token_bytes = token_byte_sizes(chars, bos)
    train_sequences = encode_docs(train_docs, stoi, bos)
    val_sequences = encode_docs(val_docs, stoi, bos)
    train_windows = build_full_eval_bank(train_sequences, cfg.train_seq_len)
    micro_bank = build_eval_bank(val_sequences, cfg.proxy_micro_num_windows, cfg.train_seq_len, cfg.proxy_seed)
    mid_bank = build_eval_bank(val_sequences, cfg.proxy_mid_num_windows, cfg.mid_seq_len, cfg.proxy_seed + 1)
    full_bank = build_full_eval_bank(val_sequences, cfg.full_seq_len)

    max_seq_len = max(cfg.train_seq_len, cfg.mid_seq_len, cfg.full_seq_len)
    model = ScalarTransformer(vocab_size=len(chars) + 1, max_seq_len=max_seq_len, spec=spec)

    print(f"model_name={spec.name}")
    print(f"model_params={len(model.params)}")
    print(
        "architecture "
        f"layers={spec.n_layer} embd={spec.n_embd} heads={spec.n_head} mlp_mult={spec.mlp_mult} "
        f"activation={spec.activation} skip_path={int(spec.use_skip_path)} "
        f"attn_scale={int(spec.use_attn_scale)} mlp_scale={int(spec.use_mlp_scale)} "
        f"resid_mix={int(spec.use_resid_mix)} q_gain={int(spec.use_q_gain)} "
        f"smear={int(spec.use_smear)} bigram={int(spec.use_bigram)}"
    )
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
        f"proxy_seed={cfg.proxy_seed} eval_every={cfg.eval_every}"
    )
    print(
        "proxy_bank_meta "
        f"train_windows={len(train_windows)} micro_bank_size={len(micro_bank)} "
        f"mid_bank_size={len(mid_bank)} full_bank_size={len(full_bank)}"
    )

    last_micro_bpb = None
    last_mid_bpb = None
    last_full_bpb = None
    best_micro_bpb = None
    best_mid_bpb = None
    eval_rows = []

    t0 = time.perf_counter()
    for step in range(1, cfg.train_steps + 1):
        window = random.choice(train_windows)
        loss, _ = window_loss_and_bytes(model, train_sequences[window.doc_idx], window.start_offset, window.seq_len, token_bytes)
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
