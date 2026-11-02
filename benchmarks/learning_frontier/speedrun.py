"""Covert-training speedrun: continue a registered model θ under one structural class of training method at a fixed FLOP
budget, and write the run's witness record (its speedrun entry).

    python speedrun.py --data DIR --start THETA.pt --method lora --update-bytes 1e6 --lr 5e-4 --out DIR --ckpt-root DIR

The start is a `train.py --save-theta` file: a first-curves dense run's weights at 25%, 50% or 100% of its steps, with their
digest (the registered θ) and held-out loss ℓ₀. Classes (`--method`), by structure:

- S0 `adamw`: every parameter, AdamW with the first curves' dense settings. The reference.
- S0 `muon`: Muon on every block matrix (Nesterov momentum 0.95, Newton-Schulz 5, the update's RMS matched to AdamW's by
  0.2 * sqrt(max(d_out, d_in)), decoupled weight decay), AdamW on the tied embedding and the norms; one learning rate for both.
- S1 `lora`: a rank-r factor pair on every matrix, the tied embedding included, with θ and the norms frozen, merged into θ at
  the end. The update is the factors at fp32 (each matrix's counted as factors or whole, whichever is smaller). `--update-bytes
  B` picks the structure: r the nearest rank to B on every matrix (at least 1); where rank 1 on every matrix is more than twice
  B, rank 1 on the last k blocks' matrices only (no embedding), k the nearest to B. `--rank` and `--lora-blocks` set it by hand.
- S2 `mezo`: MeZO, zeroth-order SPSA on every parameter: per step, the batch's loss at θ + εz and θ - εz, z ~ N(0, I) drawn
  from a seed derived from (--seed, step), and θ -= lr * g * z with g the projected gradient. The update is its log: one fp32 g
  per step (the seeds are derived), 32 bits a step plus the 64-bit seed.
- S3 `icl`: no training: θ evaluated with more context per scored token (`--icl`): sliding windows at a stride inside the
  trained context, and windows past it with RoPE extended (NTK-aware, position interpolation, or none).

The budget is FLOPs, the same for every class at a start: `--budget-tpp` tokens per non-embedding parameter at dense AdamW's
analytic FLOPs per token (`--budget-flops` sets it directly). A training class takes as many steps of `--batch` sequences as fit
in it (its FLOPs per token, plus Muon's Newton-Schulz FLOPs per step), so S1 and S2 read more tokens than S0. Every class reads
the training stream's windows in the first curves' permuted order from `--data-offset` on (default: past every window the l
base run read, so no start has seen them), and is evaluated on the held-out set at its start (ℓ₀ measured) and at every 5% of
its steps, in nats per byte. Learning rate: linear warmup over `--warmup` of the steps, cosine decay to `--min-lr` of its peak,
fresh optimizer state.

Like train.py, a run trains for at most `--chunk-minutes`, saves its state under `--ckpt-root` and exits 3; started again with
the same arguments it resumes, so `leased.sh` runs it as a chain of preemptible one-GPU leases. SIGTERM saves and exits 143.
Exit 0: `--out` holds `witness.json` (the speedrun entry: commit, configuration, seed, θ's digest and ℓ₀, analytic FLOPs,
tokens, GPU-seconds, the loss trajectory, update bits, fresh input per window, the result's digest), `curve.json` and, for S1
under `--save-update-max-mb` and for S2, the update itself (`update.pt`, `mezo-log.json`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from model import CTX, ROPE_THETA, SIZES, VOCAB, LM, Linear, counts, load_theta, matrix_shapes, theta_digest
from train import EXIT_CHUNK, EXIT_TERM, TIMED_AFTER, Corpus, adamw, git_commit, loss_of, peak_flops

CLASS = {"adamw": "S0", "muon": "S0", "lora": "S1", "mezo": "S2", "icl": "S3"}
BITS = 32
TOKEN_BITS = 16  # a token id as the corpus stores it (uint16); log2(50257) = 15.6
BASE_BATCH = 64
BASE_TPP = 8.0
#: past every window a first-curves base run read: the l run's 18,819 steps of 64 sequences, the most of the three sizes
DATA_OFFSET = max(1, round(BASE_TPP * counts(SIZES["l"])["non_embedding"] / (BASE_BATCH * CTX))) * BASE_BATCH
#: the first curves' corpus (`prep.py`): a start trained on another is refused unless --any-corpus
CORPUS_TRAIN_SHA256 = "437f108d42621f92"
MUON_SCALE = 0.2
NS_COEFFS = (3.4445, -4.7750, 2.0315)
ICL_DEFAULT = "1024:1024,1024:512,1024:256,1024:128,2048:1024:ntk,2048:1024:pi,4096:1024:ntk"


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--start", type=Path, required=True, help="a train.py --save-theta file")
    ap.add_argument("--start-sha256", default="", help="refuse a start whose weights have another digest")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt-root", type=Path, required=True)
    ap.add_argument("--method", choices=sorted(CLASS), required=True)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--budget-tpp", type=float, default=2.0, help="the budget in tokens per non-embedding parameter of dense AdamW")
    ap.add_argument("--budget-flops", type=float, default=0.0, help="the budget in analytic FLOPs (overrides --budget-tpp)")
    ap.add_argument("--update-bytes", type=float, default=0.0, help="lora: the update size to aim at (the structure rule above)")
    ap.add_argument("--rank", type=int, default=0, help="lora: the rank, by hand")
    ap.add_argument("--lora-blocks", type=int, default=0, help="lora with --rank: only the last N blocks, no embedding")
    ap.add_argument("--muon-momentum", type=float, default=0.95)
    ap.add_argument("--ns-steps", type=int, default=5)
    ap.add_argument("--mezo-eps", type=float, default=1e-3)
    ap.add_argument("--icl", default=ICL_DEFAULT, help="icl: CTX:STRIDE[:ROPE] settings, ROPE ntk, pi or none past the trained context")
    ap.add_argument("--batch", type=int, default=BASE_BATCH)
    ap.add_argument("--micro", type=int, default=0, help="sequences per forward pass (default: the whole batch)")
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--min-lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--data-offset", type=int, default=DATA_OFFSET, help="sequences of the permuted stream to skip")
    ap.add_argument("--evals", type=int, default=20)
    ap.add_argument("--chunk-minutes", type=float, default=26.0)
    ap.add_argument("--save-update-max-mb", type=float, default=16.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (a smoke test); 0: the whole budget")
    ap.add_argument("--val-windows", type=int, default=0, help="held-out windows to evaluate (a smoke test); 0: all")
    ap.add_argument("--any-corpus", action="store_true", help="allow a corpus other than the first curves' (a smoke test)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--commit", default="", help="the source commit to record (default: git rev-parse HEAD)")
    a = ap.parse_args(argv)
    if a.method != "icl" and a.lr <= 0:
        ap.error(f"--method {a.method} needs --lr")
    if a.method == "lora" and not (a.update_bytes > 0) != (a.rank > 0):
        ap.error("--method lora takes one of --update-bytes and --rank")
    if a.method != "lora" and (a.update_bytes or a.rank or a.lora_blocks):
        ap.error("--update-bytes, --rank and --lora-blocks are for --method lora")
    return a


# ---------------------------------------------------------------------------------------------------------------- structure

def forward_flops_per_token(size, ctx: int = CTX, vocab: int = VOCAB) -> float:
    """One forward pass: 2 per matrix entry (the tied head counted once, the lookup a gather) plus causal attention, QK^T and
    AV over half the context on average, 2 * ctx * d per layer."""
    c = counts(size, ctx=ctx, vocab=vocab)
    return 2 * (c["matrices"] + c["embedding"]) + 2 * size.n_layer * ctx * size.d_model


def lora_plan(size, target_bytes: float, rank: int = 0, last_blocks: int = 0, vocab: int = VOCAB) -> dict:
    """The rank, the blocks adapted and whether the embedding is: by hand (`rank`, `last_blocks`), or by the module doc's rule."""
    layer = matrix_shapes(size, vocab)[:7]
    per_block = sum(i + o for i, o in layer)
    per_rank_all = size.n_layer * per_block + vocab + size.d_model
    if rank:
        k = last_blocks or size.n_layer
        return {"rank": rank, "blocks": list(range(size.n_layer - k, size.n_layer)), "embed": not last_blocks, "rule": "by hand"}
    if BITS / 8 * per_rank_all > 2 * target_bytes:
        k = min(size.n_layer, max(1, round(target_bytes / (BITS / 8 * per_block))))
        return {"rank": 1, "blocks": list(range(size.n_layer - k, size.n_layer)), "embed": False,
                "rule": f"rank 1 on every matrix is {BITS / 8 * per_rank_all:.0f} bytes, over twice the target: the last {k} blocks"}
    r = max(1, round(target_bytes / (BITS / 8 * per_rank_all)))
    return {"rank": r, "blocks": list(range(size.n_layer)), "embed": True, "rule": "the nearest rank on every matrix"}


def lora_shapes(size, plan: dict, vocab: int = VOCAB) -> list[tuple[int, int]]:
    layer = matrix_shapes(size, vocab)[:7]
    return [s for _ in plan["blocks"] for s in layer] + ([(vocab, size.d_model)] if plan["embed"] else [])


def lora_update_values(size, plan: dict, vocab: int = VOCAB) -> int:
    r = plan["rank"]
    return sum(min(r * (i + o), i * o) for i, o in lora_shapes(size, plan, vocab))


def lora_flops_per_token(size, plan: dict, ctx: int = CTX, vocab: int = VOCAB) -> float:
    """Blocks below the lowest adapted one (none, when the embedding is) run forward only: 2 per entry and 2 * ctx * d of
    attention. From it up, a frozen matrix costs 4 per entry (no weight gradient), attention 6 * ctx * d, and each factor entry
    6; the tied head 4 per entry, plus its factor pair's 6 per entry when the embedding is adapted. train.py's per-window low
    rank is the case of every block and the embedding."""
    d, r = size.d_model, plan["rank"]
    layer = matrix_shapes(size, vocab)[:7]
    n_block, f_block = sum(i * o for i, o in layer), sum(i + o for i, o in layer)
    first = 0 if plan["embed"] else min(plan["blocks"])
    total = 4 * vocab * d + (6 * r * (vocab + d) if plan["embed"] else 0)
    for i in range(size.n_layer):
        if i < first:
            total += 2 * n_block + 2 * ctx * d
        else:
            total += 4 * n_block + 6 * ctx * d + (6 * r * f_block if i in plan["blocks"] else 0)
    return total


def ns_flops(shape: tuple[int, ...], steps: int) -> float:
    """Newton-Schulz on an m x n matrix, a = min(m, n), b = max: per step X X^T (2 a^2 b), A @ A (2 a^3) and B @ X (2 a^2 b)."""
    a, b = min(shape), max(shape)
    return steps * (4 * a * a * b + 2 * a ** 3)


def attach_lora(model: LM, plan: dict, seed: int) -> list[tuple[torch.Tensor, nn.Parameter, nn.Parameter, float]]:
    """Freeze θ (every matrix, the embedding and the norms) and give the plan's matrices a factor pair, A ~ N(0, 1/d_in) from a
    generator seeded by `seed`, B = 0; return (frozen matrix, A, B, scale) for each pair."""
    for p in model.parameters():
        p.requires_grad_(False)
    pairs = []
    for i in plan["blocks"]:
        for m in model.blocks[i].modules():
            if isinstance(m, Linear):
                m.attach(plan["rank"])
                pairs.append((m.weight, m.lora_a, m.lora_b, m.scale))
    if plan["embed"]:
        d, r = model.size.d_model, plan["rank"]
        model.rank, model.emb_scale = r, math.sqrt(d / r)
        model.emb_a = nn.Parameter(torch.zeros(r, d, device=model.embed.device))
        model.emb_b = nn.Parameter(torch.zeros(model.vocab, r, device=model.embed.device))
        pairs.append((model.embed, model.emb_a, model.emb_b, model.emb_scale))
    g = torch.Generator(device=model.embed.device).manual_seed(seed * 1_000_003)
    with torch.no_grad():
        for _, a, b, _ in pairs:
            a.copy_(torch.randn(a.shape, generator=g, device=a.device) / math.sqrt(a.shape[1]))
            b.zero_()
    return pairs


@torch.no_grad()
def merged_state(model: LM, pairs: list) -> dict[str, torch.Tensor]:
    """θ' = θ + s * B @ A on every adapted matrix, as a plain model's state (the factors folded in, their keys dropped)."""
    merged = {id(w): (w + s * (b @ a)).float() for w, a, b, s in pairs}
    out = {}
    for k, v in model.state_dict(keep_vars=True).items():
        if k.endswith(("lora_a", "lora_b")) or k in ("emb_a", "emb_b"):
            continue
        out[k] = merged.get(id(v), v.detach().float()).detach().cpu().clone()
    return out


# --------------------------------------------------------------------------------------------------------------- optimizers

def newton_schulz(g: torch.Tensor, steps: int) -> torch.Tensor:
    """The orthogonal factor of g, approximately (Muon's quintic iteration, in bf16)."""
    a, b, c = NS_COEFFS
    x = g.bfloat16()
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.mT
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        m = x @ x.mT
        x = a * x + (b * m + c * m @ m) @ x
    return x.mT if tall else x


class Muon(torch.optim.Optimizer):
    """Nesterov momentum, orthogonalized by Newton-Schulz, scaled to AdamW's update RMS (0.2 * sqrt(max(d_out, d_in))), with
    decoupled weight decay."""

    def __init__(self, params, lr: float, momentum: float, wd: float, ns_steps: int):
        super().__init__(params, {"lr": lr, "momentum": momentum, "weight_decay": wd, "ns_steps": ns_steps})

    @torch.no_grad()
    def step(self) -> None:
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("momentum_buffer", torch.zeros_like(p))
                buf.mul_(g["momentum"]).add_(p.grad)
                o = newton_schulz(p.grad.add(buf, alpha=g["momentum"]), g["ns_steps"])
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(o.to(p.dtype), alpha=-g["lr"] * MUON_SCALE * math.sqrt(max(p.shape)))


def mezo_seed(seed: int, step: int) -> int:
    return int.from_bytes(hashlib.sha256(f"mezo:{seed}:{step}".encode()).digest()[:8], "little") & (2 ** 63 - 1)


@torch.no_grad()
def perturb(params: list[nn.Parameter], seed: int, scale: float) -> None:
    """θ += scale * z, z ~ N(0, I) drawn parameter by parameter from one generator seeded by `seed`."""
    g = torch.Generator(device=params[0].device).manual_seed(seed)
    for p in params:
        p.add_(torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype), alpha=scale)


# --------------------------------------------------------------------------------------------------------------- evaluation

def rope_tables(size, n: int, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin for n positions; past the trained context, `ntk` raises the base by s^(hd / (hd - 2)), `pi` divides positions
    by s (s = n / CTX), `none` extrapolates."""
    hd = size.d_model // size.n_head
    s = max(1.0, n / CTX)
    base = ROPE_THETA * (s ** (hd / (hd - 2)) if mode == "ntk" else 1.0)
    freqs = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    pos = torch.arange(n, dtype=torch.float32) / (s if mode == "pi" else 1.0)
    angles = torch.outer(pos, freqs)
    return angles.cos(), angles.sin()


def windows(targets: int, ctx: int, stride: int) -> list[tuple[int, int]]:
    """(begin, first scored position) of each window of `ctx` tokens: window w starts at w * stride (the last one at
    targets - ctx) and scores the targets no earlier window scored, so every target is scored once, with at least
    ctx - stride tokens of context after the first window."""
    if ctx > targets or not 0 < stride <= ctx:
        raise SystemExit(f"speedrun: windows of {ctx} at stride {stride} don't fit {targets} held-out targets")
    out, prev_end, begin = [], 0, 0
    while prev_end < targets:
        b = min(begin, targets - ctx)
        out.append((b, prev_end - b))
        prev_end, begin = b + ctx, begin + stride
    return out


@torch.no_grad()
def evaluate(model: nn.Module, val: np.ndarray, targets: int, val_bytes: int | None, device: str, micro: int,
             ctx: int = CTX, stride: int = CTX, rope: str = "none", size=None) -> dict:
    """Held-out loss with every target scored once (`windows`); at ctx = stride = CTX, train.py's evaluation."""
    wins = windows(targets, ctx, stride)
    restore = None
    if ctx > model.cos.shape[0]:
        restore = (model.cos, model.sin)
        cos, sin = rope_tables(size, ctx, rope)
        model.cos, model.sin = cos.to(device), sin.to(device)
    v = torch.from_numpy(val[: targets + 1].astype(np.int64))
    pos = torch.arange(ctx)
    total, scored = torch.zeros((), dtype=torch.float64, device=device), 0
    per = max(1, micro * CTX // ctx)
    dev = device.split(":")[0]
    try:
        for i in range(0, len(wins), per):
            chunk = wins[i: i + per]
            rows = torch.stack([v[b: b + ctx + 1] for b, _ in chunk]).to(device)
            keep = torch.stack([pos >= f for _, f in chunk]).to(device)
            with torch.autocast(dev, dtype=torch.bfloat16):
                nll = loss_of(model, rows[:, :-1], rows[:, 1:], "none")
            total += (nll.view(len(chunk), ctx) * keep).sum().double()
            scored += int(keep.sum())
    finally:
        if restore:
            model.cos, model.sin = restore
    if scored != targets:
        raise SystemExit(f"speedrun: scored {scored} held-out targets, expected {targets}")
    nll = float(total)
    return {"val_nats_per_byte": nll / val_bytes if val_bytes else float("nan"), "val_nats_per_token": nll / targets,
            "ctx": ctx, "stride": stride, "rope": rope if ctx > CTX else "trained", "windows": len(wins), "targets": targets}


# ------------------------------------------------------------------------------------------------------------------ the run

class Run:
    def __init__(self, a: argparse.Namespace, theta: dict, meta: dict):
        self.a, self.meta = a, meta
        self.size = SIZES[meta["size"]]
        self.c = counts(self.size)
        self.model = LM(self.size).to(a.device)
        self.model.load_state_dict({k: v.to(a.device) for k, v in theta.items()})
        self.pairs, self.plan, self.opts = [], None, []
        fused = a.device.startswith("cuda")
        tokens_per_step = a.batch * CTX
        self.opt_flops_per_step = 0.0
        m = a.method
        if m == "adamw":
            self.flops_per_token = self.c["dense_flops_per_token"]
            self.opts = [adamw(list(self.model.parameters()), a, fused)]
        elif m == "muon":
            self.flops_per_token = self.c["dense_flops_per_token"]
            mats = [p for n, p in self.model.named_parameters() if p.ndim == 2 and n != "embed"]
            rest = [p for n, p in self.model.named_parameters() if not (p.ndim == 2 and n != "embed")]
            self.opts = [Muon(mats, a.lr, a.muon_momentum, a.wd, a.ns_steps), adamw(rest, a, fused)]
            self.opt_flops_per_step = sum(ns_flops(tuple(p.shape), a.ns_steps) for p in mats)
        elif m == "lora":
            self.plan = lora_plan(self.size, a.update_bytes, a.rank, a.lora_blocks)
            self.pairs = attach_lora(self.model, self.plan, a.seed)
            self.flops_per_token = lora_flops_per_token(self.size, self.plan)
            self.opts = [adamw([p for p in self.model.parameters() if p.requires_grad], a, fused)]
        elif m == "mezo":
            self.flops_per_token = 2 * forward_flops_per_token(self.size)
        else:
            self.flops_per_token = 0.0
        self.budget = a.budget_flops or a.budget_tpp * self.c["non_embedding"] * self.c["dense_flops_per_token"]
        self.flops_per_step = self.flops_per_token * tokens_per_step + self.opt_flops_per_step
        self.n_steps = int(self.budget // self.flops_per_step) if self.flops_per_step else 0
        self.micro = a.micro or a.batch
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.compiled = self.model if a.no_compile or m == "icl" else torch.compile(self.model)
        self.step, self.curve, self.ema, self.mezo_log = 0, [], None, []
        self.timed_steps, self.timed_s, self.wall_s, self.eval_s = 0, 0.0, 0.0, 0.0
        self.starts, self.train_verified, self.commits = 0, False, []

    def key(self, corpus_sha: str) -> str:
        skip = {"out", "ckpt_root", "chunk_minutes", "device", "no_compile", "data", "commit", "start", "save_update_max_mb"}
        spec = {k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(vars(self.a).items()) if k not in skip}
        return hashlib.sha256(json.dumps([spec, self.meta["theta_sha256"], corpus_sha], sort_keys=True).encode()).hexdigest()[:20]

    def state(self) -> dict:
        return {"step": self.step, "curve": self.curve, "model": self.model.state_dict(),
                "opts": [o.state_dict() for o in self.opts], "mezo_log": self.mezo_log, "timed_steps": self.timed_steps,
                "timed_s": self.timed_s, "wall_s": self.wall_s, "eval_s": self.eval_s, "starts": self.starts,
                "train_verified": self.train_verified, "commits": self.commits, "ema": self.ema}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state(), path / "state.pt.tmp")
        (path / "state.pt.tmp").rename(path / "state.pt")

    def load(self, path: Path) -> bool:
        if not (path / "state.pt").exists():
            return False
        s = torch.load(path / "state.pt", map_location=self.a.device, weights_only=False)
        self.model.load_state_dict(s["model"])
        for o, os_ in zip(self.opts, s["opts"]):
            o.load_state_dict(os_)
        for k in ("step", "curve", "mezo_log", "timed_steps", "timed_s", "wall_s", "eval_s", "starts", "train_verified",
                  "commits", "ema"):
            setattr(self, k, s[k])
        return True

    def eval_steps(self) -> list[int]:
        n, e = self.n_steps, self.a.evals
        return sorted({max(1, round(n * k / e)) for k in range(1, e + 1)})

    def lr(self, step: int) -> float:
        a, n = self.a, self.n_steps
        warm = max(1, round(a.warmup * n))
        if step < warm:
            return a.lr * (step + 1) / warm
        frac = (step - warm) / max(1, n - warm)
        return a.lr * (a.min_lr + (1 - a.min_lr) * 0.5 * (1 + math.cos(math.pi * frac)))

    def batch_loss(self, corpus: Corpus, idx: np.ndarray, backward: bool) -> torch.Tensor:
        """The batch's mean loss, on the device (no host sync, so the next batch's copy overlaps the step)."""
        dev = self.a.device.split(":")[0]
        losses = []
        for i in range(0, len(idx), self.micro):
            x, y = corpus.batch(idx[i: i + self.micro], self.a.device)
            with torch.autocast(dev, dtype=torch.bfloat16):
                loss = loss_of(self.compiled, x, y) * (len(x) / len(idx))
            if backward:
                loss.backward()
            losses.append(loss.detach())
        return torch.stack(losses).sum()

    def train_step(self, corpus: Corpus, order: np.ndarray) -> torch.Tensor:
        a, b = self.a, self.a.batch
        idx = order[self.step * b: (self.step + 1) * b]
        lr = self.lr(self.step)
        if a.method == "mezo":
            seed, eps = mezo_seed(a.seed, self.step), a.mezo_eps
            with torch.no_grad():
                perturb(self.params, seed, eps)
                plus = float(self.batch_loss(corpus, idx, False))
                perturb(self.params, seed, -2 * eps)
                minus = float(self.batch_loss(corpus, idx, False))
                perturb(self.params, seed, eps)
                g = (plus - minus) / (2 * eps)
                perturb(self.params, seed, -lr * g)
            self.mezo_log.append(g)
            loss = torch.tensor((plus + minus) / 2)
        else:
            for o in self.opts:
                for grp in o.param_groups:
                    grp["lr"] = lr
            loss = self.batch_loss(corpus, idx, True)
            torch.nn.utils.clip_grad_norm_(self.params, a.clip)
            for o in self.opts:
                o.step()
                o.zero_grad(set_to_none=True)
        self.step += 1
        return loss

    def tally(self, step: int) -> dict:
        tokens = step * self.a.batch * CTX
        return {"step": step, "tokens": tokens, "flops": step * self.flops_per_step,
                "budget_fraction": step * self.flops_per_step / self.budget}

    def update(self) -> dict:
        """The update's structure and size in bits (the bound's input), and what is kept of it."""
        a, m = self.a, self.a.method
        if m in ("adamw", "muon"):
            return {"values": self.c["total"], "bits": BITS * self.c["total"], "what": "every parameter at fp32"}
        if m == "lora":
            v = lora_update_values(self.size, self.plan)
            return {"values": v, "bits": BITS * v, "bytes": BITS * v / 8, "target_bytes": a.update_bytes or None,
                    "plan": self.plan, "what": "each adapted matrix's factors at fp32, or the matrix whole where smaller"}
        if m == "mezo":
            return {"values": len(self.mezo_log), "bits": BITS * len(self.mezo_log) + 64,
                    "what": "one fp32 projected gradient per step and the 64-bit seed; z and the schedule are derived"}
        return {"values": 0, "bits": 0, "what": "none: θ is not trained"}


def icl(a: argparse.Namespace, run: Run, corpus: Corpus, start: dict, commit: str, base: dict) -> int:
    """S3: θ at each --icl setting, with the forward FLOPs per scored target and the context it reads."""
    windows_n = a.val_windows or corpus.manifest["val"]["targets"] // CTX
    targets = windows_n * CTX
    val_bytes = corpus.manifest["val"]["val_bytes"] if targets == corpus.manifest["val"]["targets"] else None
    points = []
    for spec in a.icl.split(","):
        parts = spec.split(":")
        ctx, stride, rope = int(parts[0]), int(parts[1]), (parts[2] if len(parts) > 2 else "none")
        t0 = time.perf_counter()
        r = evaluate(run.model, corpus.val, targets, val_bytes, a.device, run.micro, ctx, stride, rope, run.size)
        dt = time.perf_counter() - t0
        fwd = r["windows"] * ctx * forward_flops_per_token(run.size, ctx) / targets
        points.append(r | {"forward_flops_per_target": fwd,
                           "inference_multiplier": fwd / forward_flops_per_token(run.size, CTX),
                           "least_context": ctx - stride if stride < ctx else 0,
                           "fresh_input_bits_per_window": TOKEN_BITS * ctx, "seconds": dt})
        print(json.dumps({k: points[-1][k] for k in ("ctx", "stride", "rope", "val_nats_per_byte", "inference_multiplier")}),
              flush=True)
    (a.out / "curve.json").write_text(json.dumps(points, indent=1) + "\n")
    witness = base | {"steps": 0, "tokens": 0, "flops": 0.0, "trajectory": points,
                      "update": run.update(), "update_bits": 0, "theta_out_sha256": start["theta_sha256"],
                      "min_val_nats_per_byte": min(p["val_nats_per_byte"] for p in points),
                      "gpu_seconds": {"wall": sum(p["seconds"] for p in points), "eval": sum(p["seconds"] for p in points)}}
    (a.out / "witness.json").write_text(json.dumps(witness, indent=1) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    a = parse(argv)
    t_start = time.perf_counter()
    term = {"set": False}
    signal.signal(signal.SIGTERM, lambda *_: term.update(set=True))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    theta, start = load_theta(a.start, a.start_sha256)
    manifest = json.loads((a.data / "manifest.json").read_text())
    if not a.any_corpus and not (manifest["train"]["sha256"].startswith(CORPUS_TRAIN_SHA256)
                                 and start.get("corpus_train_sha256", "").startswith(CORPUS_TRAIN_SHA256)):
        raise SystemExit(f"speedrun: the corpus or the start's is not the first curves' ({CORPUS_TRAIN_SHA256}...)")
    torch.manual_seed(a.seed)
    run = Run(a, theta, start)
    key = run.key(manifest["train"]["sha256"])
    ckpt = a.ckpt_root / key
    resumed = run.load(ckpt)
    corpus = Corpus(a.data, verify_train=not run.train_verified and a.method != "icl")
    run.train_verified = True
    run.starts += 1
    commit = a.commit or git_commit()
    if run.commits[-1:] != [commit]:
        run.commits.append(commit)
    n_seq_needed = a.data_offset + run.n_steps * a.batch
    if n_seq_needed > corpus.n_seq:
        raise SystemExit(f"speedrun: the run reads sequences up to {n_seq_needed}, the corpus has {corpus.n_seq}")
    perm = np.random.default_rng(a.data_seed).permutation(corpus.n_seq)
    order = perm[a.data_offset: n_seq_needed]
    peak = peak_flops()
    gpu = torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu"
    base = {"schema": "covert-speedrun/witness/v1", "class": CLASS[a.method], "method": a.method, "key": key,
            "commit": commit, "commits": run.commits,
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
            "seed": a.seed, "lr": a.lr, "size": start["size"],
            "start": {"file": str(a.start), "theta_sha256": start["theta_sha256"], "fraction": start.get("fraction"),
                      "step": start.get("step"), "n_steps": start.get("n_steps"), "lr": start.get("lr"),
                      "val_nats_per_byte_recorded": start.get("val_nats_per_byte"), "key": start.get("key"),
                      "commit": start.get("commit")},
            "budget_flops": run.budget, "budget_tpp": None if a.budget_flops else a.budget_tpp,
            "flops_per_token": run.flops_per_token, "optimizer_flops_per_step": run.opt_flops_per_step,
            "flops_per_step": run.flops_per_step, "n_steps": run.n_steps, "counts": run.c,
            "fresh_input_bits_per_window": TOKEN_BITS * CTX,
            "data": {"rule": "the training stream cut into ctx + 1 token windows at stride ctx, permuted by numpy "
                             "default_rng(data_seed).permutation (train.py's order); step t reads permuted windows "
                             "[offset + t * batch, offset + (t + 1) * batch)", "data_seed": a.data_seed,
                     "offset": a.data_offset, "sequences": int(corpus.n_seq),
                     "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(), "corpus": manifest},
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": gpu, "peak_flop_per_second": peak}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps(base | {"starts": run.starts, "resumed_at_step": run.step if resumed else None},
                                                  indent=1) + "\n")
    print(f"speedrun: {CLASS[a.method]} {a.method} size={start['size']} start=p{round(100 * (start.get('fraction') or 0)):03d} "
          f"lr={a.lr:g} steps={run.n_steps} budget={run.budget:.3e} key={key} resume={run.step if resumed else 0}", flush=True)
    if a.method == "icl":
        return icl(a, run, corpus, start, commit, base)

    windows_n = a.val_windows or corpus.manifest["val"]["targets"] // CTX
    targets = windows_n * CTX
    val_bytes = corpus.manifest["val"]["val_bytes"] if targets == corpus.manifest["val"]["targets"] else None
    n_steps = min(run.n_steps, a.max_steps) if a.max_steps else run.n_steps
    marks = [m for m in run.eval_steps() if m <= n_steps] or [n_steps]

    def sync() -> None:
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()

    def record(step: int) -> dict:
        sync()
        t0 = time.perf_counter()
        r = evaluate(run.compiled, corpus.val, targets, val_bytes, a.device, run.micro)
        run.eval_s += time.perf_counter() - t0
        spt = run.timed_s / run.timed_steps if run.timed_steps else None
        point = run.tally(step) | {"val_nats_per_byte": r["val_nats_per_byte"], "val_nats_per_token": r["val_nats_per_token"],
                                   "lr": run.lr(max(0, step - 1)) if step else 0.0,
                                   "train_loss_ema": None if run.ema is None else float(run.ema), "seconds_per_step": spt,
                                   "tokens_per_second": a.batch * CTX / spt if spt else None,
                                   "mfu": run.flops_per_step / spt / peak if spt else None}
        run.curve.append(point)
        (a.out / "curve.json").write_text(json.dumps(run.curve, indent=1) + "\n")
        print(json.dumps({k: point[k] for k in ("step", "budget_fraction", "val_nats_per_byte", "train_loss_ema")}), flush=True)
        return point

    if not run.curve:
        p0 = record(0)
        want = start.get("val_nats_per_byte")
        if want is not None and val_bytes and abs(p0["val_nats_per_byte"] - want) > 2e-3:
            raise SystemExit(f"speedrun: the start evaluates to {p0['val_nats_per_byte']:.4f}, its record says {want:.4f}")
    seg_t0, seg_steps, first = None, 0, run.step
    while run.step < n_steps:
        loss = run.train_step(corpus, order)
        run.ema = loss if run.ema is None else 0.98 * run.ema + 0.02 * loss.to(run.ema.device)
        if (run.step % 25 == 0 or run.step in marks) and not math.isfinite(float(loss)):
            run.curve.append({"step": run.step, "diverged": True, "train_loss": float(loss)})
            (a.out / "curve.json").write_text(json.dumps(run.curve, indent=1) + "\n")
            print(f"speedrun: diverged at step {run.step}", flush=True)
            break
        if run.step - first == TIMED_AFTER:
            sync()
            seg_t0, seg_steps = time.perf_counter(), 0
        elif seg_t0 is not None:
            seg_steps += 1
        if run.step in marks:
            sync()
            if seg_t0 is not None:
                run.timed_s += time.perf_counter() - seg_t0
                run.timed_steps += seg_steps
            record(run.step)
            if run.step < n_steps:
                run.save(ckpt)
            if seg_t0 is not None:
                sync()
                seg_t0, seg_steps = time.perf_counter(), 0
        if (term["set"] or (time.perf_counter() - t_start) / 60 > a.chunk_minutes) and run.step < n_steps:
            if seg_t0 is not None:
                sync()
                run.timed_s += time.perf_counter() - seg_t0
                run.timed_steps += seg_steps
            run.wall_s += time.perf_counter() - t_start
            run.save(ckpt)
            print(f"speedrun: {'SIGTERM' if term['set'] else 'chunk ended'} at step {run.step}; state saved", flush=True)
            return EXIT_TERM if term["set"] else EXIT_CHUNK

    diverged = bool(run.curve and run.curve[-1].get("diverged"))
    final_state = merged_state(run.model, run.pairs) if run.pairs else {k: v.detach().float().cpu()
                                                                         for k, v in run.model.state_dict().items()}
    extra = {}
    if run.pairs and not diverged:
        check = LM(run.size).to(a.device)
        check.load_state_dict({k: v.to(a.device) for k, v in final_state.items()})
        extra["merged_val_nats_per_byte"] = evaluate(check, corpus.val, targets, val_bytes, a.device, run.micro)["val_nats_per_byte"]
        upd = run.update()
        if upd["bits"] / 8 / 1e6 <= a.save_update_max_mb:
            torch.save({"plan": run.plan, "factors": {f"{i}": (A.detach().cpu(), B.detach().cpu(), s)
                                                      for i, (_, A, B, s) in enumerate(run.pairs)}}, a.out / "update.pt")
            extra["update_file"] = "update.pt"
    if a.method == "mezo":
        (a.out / "mezo-log.json").write_text(json.dumps({"seed": a.seed, "eps": a.mezo_eps, "g": run.mezo_log}) + "\n")
        extra["update_file"] = "mezo-log.json"
    run.wall_s += time.perf_counter() - t_start
    good = [p for p in run.curve if not p.get("diverged")]
    spt = run.timed_s / run.timed_steps if run.timed_steps else None
    upd = run.update()
    witness = base | {"commits": run.commits, "steps": run.step, "complete": run.step >= run.n_steps, "diverged": diverged,
                      "tokens": run.step * a.batch * CTX, "flops": run.step * run.flops_per_step,
                      "trajectory": run.curve, "l0_measured": good[0]["val_nats_per_byte"] if good else None,
                      "final_val_nats_per_byte": good[-1]["val_nats_per_byte"] if good else None,
                      "min_val_nats_per_byte": min(p["val_nats_per_byte"] for p in good) if good else None,
                      "update": upd, "update_bits": upd["bits"], "theta_out_sha256": theta_digest(final_state),
                      "gpu_seconds": {"wall": run.wall_s, "train_timed": run.timed_s, "eval": run.eval_s,
                                      "timed_steps": run.timed_steps},
                      "seconds_per_step": spt, "tokens_per_second": a.batch * CTX / spt if spt else None,
                      "mfu": run.flops_per_step / spt / peak if spt else None, "starts": run.starts,
                      "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if a.device.startswith("cuda") else None
                      } | extra
    (a.out / "witness.json").write_text(json.dumps(witness, indent=1) + "\n")
    if witness["complete"] or diverged:
        shutil.rmtree(ckpt, ignore_errors=True)
    print(json.dumps({k: witness[k] for k in ("class", "method", "size", "lr", "steps", "min_val_nats_per_byte", "update_bits")}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
