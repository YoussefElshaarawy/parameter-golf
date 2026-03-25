from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn


DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class EvalWindow:
    shard_idx: int
    start_offset: int
    seq_len: int


@dataclass
class Config:
    seed: int = int(os.environ.get("SEED", "42"))
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    data_path: str = os.environ.get(
        "DATA_PATH",
        str(repo_root() / "data" / "datasets" / "fineweb10B_sp1024"),
    )
    tokenizer_path: str = os.environ.get(
        "TOKENIZER_PATH",
        str(repo_root() / "data" / "tokenizers" / "fineweb_1024_bpe.model"),
    )
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", "1024"))

    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", "64"))
    eval_seq_len: int = int(os.environ.get("EVAL_SEQ_LEN", "64"))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", "1024"))
    train_steps: int = int(os.environ.get("TRAIN_STEPS", "1000"))
    eval_every: int = int(os.environ.get("EVAL_EVERY", "0"))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", "50"))

    proxy_micro_num_windows: int = int(os.environ.get("PROXY_MICRO_NUM_WINDOWS", "8"))
    proxy_mid_num_windows: int = int(os.environ.get("PROXY_MID_NUM_WINDOWS", "32"))
    proxy_micro_every: int = int(os.environ.get("PROXY_MICRO_EVERY", "50"))
    proxy_mid_every: int = int(os.environ.get("PROXY_MID_EVERY", "200"))
    proxy_seed: int = int(os.environ.get("PROXY_SEED", "12345"))

    n_layer: int = int(os.environ.get("NUM_LAYERS", "1"))
    n_embd: int = int(os.environ.get("MODEL_DIM", "8"))
    n_head: int = int(os.environ.get("NUM_HEADS", "2"))
    mlp_mult: float = float(os.environ.get("MLP_MULT", "1.5"))
    emb_rank: int = int(os.environ.get("EMB_RANK", "2"))
    dropout: float = float(os.environ.get("DROPOUT", "0.0"))
    lr: float = float(os.environ.get("LR", "1e-3"))
    weight_decay: float = float(os.environ.get("WEIGHT_DECAY", "0.0"))
    grad_clip: float = float(os.environ.get("GRAD_CLIP", "1.0"))
    warmdown_steps: int = int(os.environ.get("WARMDOWN_STEPS", "200"))

    @property
    def train_files(self) -> list[Path]:
        return sorted(Path(self.data_path).glob("fineweb_train_*.bin"))

    @property
    def val_files(self) -> list[Path]:
        return sorted(Path(self.data_path).glob("fineweb_val_*.bin"))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.v = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, dim = x.shape
        q = self.q(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(cfg.n_embd * cfg.mlp_mult)
        self.fc = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.mlp_norm = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.token_factors = nn.Embedding(cfg.vocab_size, cfg.emb_rank)
        self.token_proj = nn.Linear(cfg.emb_rank, cfg.n_embd, bias=False)
        self.pos_emb = nn.Embedding(cfg.eval_seq_len, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> Tensor:
        _, seq_len = idx.shape
        pos = torch.arange(seq_len, device=idx.device)
        x = self.token_proj(self.token_factors(idx)) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = torch.einsum("btd,dr,vr->btv", x, self.token_proj.weight, self.token_factors.weight)
        if targets is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def load_data_shard(path: Path) -> Tensor:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != DATAFILE_MAGIC or int(header[1]) != DATAFILE_VERSION:
        raise ValueError(f"Bad shard header: {path}")
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=256 * np.dtype("<i4").itemsize)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read: {path}")
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, files: list[Path]):
        if not files:
            raise FileNotFoundError("No shard files found")
        self.files = files
        self.file_idx = 0
        self.tokens = load_data_shard(files[0])
        self.pos = 0

    def _advance(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, num_tokens: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = num_tokens
        while remaining > 0:
            available = self.tokens.numel() - self.pos
            if available <= 0:
                self._advance()
                continue
            take_n = min(available, remaining)
            chunks.append(self.tokens[self.pos : self.pos + take_n])
            self.pos += take_n
            remaining -= take_n
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


def next_batch(stream: TokenStream, cfg: Config) -> tuple[Tensor, Tensor]:
    batch_size = max(1, cfg.train_batch_tokens // cfg.train_seq_len)
    local = stream.take(batch_size * cfg.train_seq_len + 1).to(torch.long)
    x = local[:-1].reshape(batch_size, cfg.train_seq_len)
    y = local[1:].reshape(batch_size, cfg.train_seq_len)
    return x.to(cfg.device), y.to(cfg.device)


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor,
    vocab_size: int,
    device: str,
) -> tuple[Tensor, Tensor, Tensor]:
    table_size = max(int(sp.vocab_size()), vocab_size)
    base_bytes = np.zeros((table_size,), dtype=np.int16)
    has_leading_space = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(int(sp.vocab_size())):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token[token_id] = False
        if sp.is_byte(token_id):
            base_bytes[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space[token_id] = True
            piece = piece[1:]
        base_bytes[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token, dtype=torch.bool, device=device),
    )


def token_byte_count(
    prev_ids: Tensor,
    target_ids: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> Tensor:
    token_bytes = base_bytes_lut[target_ids].to(torch.int32)
    token_bytes += (
        has_leading_space_lut[target_ids] & ~is_boundary_token_lut[prev_ids]
    ).to(torch.int32)
    return token_bytes


def load_validation_shards(files: list[Path]) -> list[Tensor]:
    if not files:
        raise FileNotFoundError("No validation shards found")
    return [load_data_shard(path).contiguous() for path in files]


def build_eval_bank(
    val_shards: list[Tensor],
    num_windows: int,
    seq_len: int,
    seed: int,
) -> list[EvalWindow]:
    counts = []
    total_windows = 0
    for shard in val_shards:
        count = max(0, shard.numel() - seq_len)
        counts.append(count)
        total_windows += count
    if total_windows <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    num_windows = min(num_windows, total_windows)
    rng = random.Random(seed)
    sampled = sorted(rng.sample(range(total_windows), num_windows))
    bank: list[EvalWindow] = []
    shard_idx = 0
    shard_base = 0
    for global_idx in sampled:
        while global_idx >= shard_base + counts[shard_idx]:
            shard_base += counts[shard_idx]
            shard_idx += 1
        bank.append(
            EvalWindow(
                shard_idx=shard_idx,
                start_offset=global_idx - shard_base,
                seq_len=seq_len,
            )
        )
    return bank


def build_full_eval_bank(val_shards: list[Tensor], seq_len: int) -> list[EvalWindow]:
    bank: list[EvalWindow] = []
    for shard_idx, shard in enumerate(val_shards):
        usable = shard.numel() - 1
        num_windows = usable // seq_len
        for window_idx in range(num_windows):
            bank.append(EvalWindow(shard_idx=shard_idx, start_offset=window_idx * seq_len, seq_len=seq_len))
    if not bank:
        raise ValueError(f"Validation split is too short for full eval seq_len={seq_len}")
    return bank


@torch.no_grad()
def run_eval_bank(
    model: GPT,
    val_shards: list[Tensor],
    bank: list[EvalWindow],
    device: str,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    batch_size: int = 8,
) -> tuple[float, float]:
    if not bank:
        raise ValueError("Evaluation bank must not be empty")
    seq_len = bank[0].seq_len
    if any(window.seq_len != seq_len for window in bank):
        raise ValueError("run_eval_bank expects a single seq_len per bank")

    loss_sum = 0.0
    token_count = 0
    byte_count = 0
    model.eval()

    for batch_start in range(0, len(bank), batch_size):
        batch = bank[batch_start : batch_start + batch_size]
        x_batch = torch.empty((len(batch), seq_len), dtype=torch.long, device=device)
        y_batch = torch.empty((len(batch), seq_len), dtype=torch.long, device=device)
        for row, window in enumerate(batch):
            shard = val_shards[window.shard_idx]
            raw = shard[window.start_offset : window.start_offset + window.seq_len + 1].to(torch.long)
            x_batch[row] = raw[:-1].to(device)
            y_batch[row] = raw[1:].to(device)
        logits = model(x_batch)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_batch.reshape(-1),
            reduction="none",
        ).reshape(len(batch), seq_len)
        loss_sum += float(losses.sum().item())
        token_count += y_batch.numel()
        byte_count += int(
            token_byte_count(
                x_batch.reshape(-1),
                y_batch.reshape(-1),
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            ).sum().item()
        )

    model.train()
    avg_loss = loss_sum / token_count
    bits_per_token = avg_loss / math.log(2.0)
    tokens_per_byte = token_count / max(byte_count, 1)
    return avg_loss, bits_per_token * tokens_per_byte


def format_metric(value: float | None) -> str:
    return "na" if value is None else f"{value:.6f}"


def maybe_eval(
    enabled: bool,
    model: GPT,
    val_shards: list[Tensor],
    bank: list[EvalWindow],
    cfg: Config,
    luts: tuple[Tensor, Tensor, Tensor],
) -> tuple[float | None, float | None]:
    if not enabled:
        return None, None
    loss, bpb = run_eval_bank(model, val_shards, bank, cfg.device, *luts)
    return loss, bpb


def main() -> None:
    cfg = Config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    if int(sp.vocab_size()) != cfg.vocab_size:
        raise ValueError(f"VOCAB_SIZE mismatch: cfg={cfg.vocab_size} tokenizer={int(sp.vocab_size())}")
    if cfg.eval_seq_len < cfg.train_seq_len:
        raise ValueError("EVAL_SEQ_LEN must be >= TRAIN_SEQ_LEN")
    if not cfg.train_files or not cfg.val_files:
        raise FileNotFoundError(f"Expected fineweb shards in {cfg.data_path}")

    train_stream = TokenStream(cfg.train_files)
    val_shards = load_validation_shards(cfg.val_files)
    luts = build_sentencepiece_luts(sp, cfg.vocab_size, cfg.device)

    micro_bank = build_eval_bank(val_shards, cfg.proxy_micro_num_windows, cfg.train_seq_len, cfg.proxy_seed)
    mid_bank = build_eval_bank(val_shards, cfg.proxy_mid_num_windows, cfg.eval_seq_len, cfg.proxy_seed + 1)
    full_bank = build_full_eval_bank(val_shards, cfg.eval_seq_len)

    model = GPT(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print(f"model_params={sum(p.numel() for p in model.parameters())}")
    print(
        f"data train_shards={len(cfg.train_files)} val_shards={len(cfg.val_files)} "
        f"train_seq_len={cfg.train_seq_len} eval_seq_len={cfg.eval_seq_len}"
    )
    print(
        "proxy_config "
        f"proxy_micro_num_windows={cfg.proxy_micro_num_windows} "
        f"proxy_mid_num_windows={cfg.proxy_mid_num_windows} "
        f"proxy_micro_every={cfg.proxy_micro_every} "
        f"proxy_mid_every={cfg.proxy_mid_every} "
        f"proxy_seed={cfg.proxy_seed}"
    )
    print(
        "proxy_bank_meta "
        f"micro_seq_len={cfg.train_seq_len} "
        f"mid_seq_len={cfg.eval_seq_len} "
        f"micro_bank_size={len(micro_bank)} "
        f"mid_bank_size={len(mid_bank)} "
        f"full_bank_size={len(full_bank)} "
        f"eval_every={cfg.eval_every}"
    )
    print(
        "proxy_bank_example "
        f"micro={micro_bank[0].shard_idx}:{micro_bank[0].start_offset}:{micro_bank[0].seq_len} "
        f"mid={mid_bank[0].shard_idx}:{mid_bank[0].start_offset}:{mid_bank[0].seq_len}"
    )

    last_micro_bpb: float | None = None
    last_mid_bpb: float | None = None
    last_full_bpb: float | None = None
    best_micro_bpb: float | None = None
    best_mid_bpb: float | None = None
    eval_rows: list[dict[str, float | int | None]] = []

    start_time = time.perf_counter()
    for step in range(1, cfg.train_steps + 1):
        x, y = next_batch(train_stream, cfg)
        loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        if cfg.warmdown_steps > 0 and step > cfg.train_steps - cfg.warmdown_steps:
            lr_scale = max((cfg.train_steps - step) / max(cfg.warmdown_steps, 1), 0.0)
        else:
            lr_scale = 1.0
        optimizer.param_groups[0]["lr"] = cfg.lr * lr_scale
        optimizer.step()

        ran_eval = False
        if cfg.proxy_micro_every > 0 and step % cfg.proxy_micro_every == 0:
            _, last_micro_bpb = maybe_eval(True, model, val_shards, micro_bank, cfg, luts)
            best_micro_bpb = last_micro_bpb if best_micro_bpb is None else min(best_micro_bpb, last_micro_bpb)
            ran_eval = True
        if cfg.proxy_mid_every > 0 and step % cfg.proxy_mid_every == 0:
            _, last_mid_bpb = maybe_eval(True, model, val_shards, mid_bank, cfg, luts)
            best_mid_bpb = last_mid_bpb if best_mid_bpb is None else min(best_mid_bpb, last_mid_bpb)
            ran_eval = True
        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            _, last_full_bpb = maybe_eval(True, model, val_shards, full_bank, cfg, luts)
            ran_eval = True

        avg_step_ms = 1000.0 * (time.perf_counter() - start_time) / step
        should_log = step <= 10 or step % cfg.train_log_every == 0 or ran_eval or step == cfg.train_steps
        if should_log:
            print(
                "metrics "
                f"step={step} "
                f"train_loss={loss.item():.6f} "
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
                    "train_loss": round(float(loss.item()), 8),
                    "avg_step_ms": round(avg_step_ms, 4),
                    "proxy_bpb_micro": None if last_micro_bpb is None else round(last_micro_bpb, 8),
                    "proxy_bpb_mid": None if last_mid_bpb is None else round(last_mid_bpb, 8),
                    "full_bpb": None if last_full_bpb is None else round(last_full_bpb, 8),
                    "best_proxy_bpb_micro": None if best_micro_bpb is None else round(best_micro_bpb, 8),
                    "best_proxy_bpb_mid": None if best_mid_bpb is None else round(best_mid_bpb, 8),
                }
            )

    if last_full_bpb is None:
        _, last_full_bpb = run_eval_bank(model, val_shards, full_bank, cfg.device, *luts)

    print("final_summary_begin")
    print(f"last_proxy_bpb_micro={format_metric(last_micro_bpb)}")
    print(f"best_proxy_bpb_micro={format_metric(best_micro_bpb)}")
    print(f"last_proxy_bpb_mid={format_metric(last_mid_bpb)}")
    print(f"best_proxy_bpb_mid={format_metric(best_mid_bpb)}")
    print(f"final_full_bpb={format_metric(last_full_bpb)}")
    print(f"correlation_rows_json={çjson.dumps(eval_rows, separators=(',', ':'))}")
    print("final_summary_end")


if __name__ == "__main__":
    main()
