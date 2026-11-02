"""Island `ref-data` (class `reference`, no cap on the update): ref-optim's train.py (cursor/te-ref-optim-4207 at 0788f9d17e6:
every optimizer knob a flag, the seed's shape by default) with the data it trains on and their order as the variable. Every
FLOP spent choosing data is in the step log. The launcher runs `islands/reference/train.py`, a symlink to this file.

Optimizer (ref-optim's): Muon or AdamW on the block matrices, AdamW on the embedding and norms, `--sched linear|wsd|cosine|const`
with warmup and `--final-frac`, EMA (`--ema-frac`) chosen against raw weights on held-back training-stream windows (priced
forwards), snapshots at `--snap-at`, and `--forks` cooldown branches (`--fork-len` of the steps before each, the rate linearly
toward 0 on windows after the plan, snapshotted, then rolled back; their passes stay in the step log).

What trains (default every tensor): `--freeze-embed`, `--freeze-norms`, `--top-blocks K` (only the top K blocks' matrices
and gains and the final norm, the embedding frozen, so the backward stops below block L-K) and `--block-lr-decay D` (block
i at D^(L-1-i) of its rate). `--class` is the Delta bin the submission declares (default reference).

Data (all off by default: TrainData's stream permuted by `--data-seed`, the seed's):

- `--windows random|fresh|seen`: the 1024-token windows drawn. `fresh` drops the windows the start read (its provenance's
  step x batch first windows of the seed-0 order, `starts.py`), `seen` keeps only those. `--fresh` (ref-optim's) is the
  start's own order past what it read.
- `--short-ctx C --short-frac F`: the first F of the steps at context C (C divides 1024; the same tokens per step), priced at
  that context; the rest at 1024.
- `--score-len P --keep K --select drop-high|drop-low|mid`: θ scores a pool of windows by its mean loss on each window's first
  P tokens (a `forward` pass at context P, priced) and K of the pool is kept.
- `--order random|easy-first|hard-first`: the kept windows' order by θ's score (needs `--score-len`).
- `--trunc Q`: in each micro-batch the Q of the sequences with the highest loss get no gradient (loss truncation; no extra
  FLOPs, their pass is still priced).

E is the trained tensors in full (`prepare.encode_full`). It runs inside the harness (candidate.sh's script mode), and
since it reads windows by id rather than through `TrainData.next`, it marks its steps itself
(`hooks.step()`): each chunk of θ's scoring pass, each training step with its snapshots and forks, and the final
validation and submit.

    launch.py reference m100 --island ref-data --note "MECHANISM ..." -- [--mat-lr 0.004] [--top-blocks 2] [--windows fresh]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from te_core import hooks

import prepare

KLASS = "reference"
CTX = prepare.EVAL_CTX


def newton_schulz(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if tall else x).to(g.dtype)


def ns_flops(shape: tuple[int, int], steps: int = 5) -> int:
    m, n = min(shape), max(shape)
    return steps * (4 * m * m * n + 2 * m ** 3)


class Muon(torch.optim.Optimizer):
    """Muon: momentum, Newton-Schulz orthogonalized. `scale` "shape" is the seed's sqrt(max(1, rows/cols)); "rms" is
    0.2 sqrt(max(rows, cols)), an update RMS of 0.2 like AdamW's, so its rate is on AdamW's scale."""

    def __init__(self, params, lr: float, momentum: float = 0.95, nesterov: bool = True, wd: float = 0.0,
                 scale: str = "shape", ns_steps: int = 5):
        super().__init__(params, {"lr": lr, "momentum": momentum, "nesterov": nesterov, "wd": wd, "scale": scale,
                                  "ns_steps": ns_steps})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state.setdefault(p, {"buf": torch.zeros_like(p)})
                buf = st["buf"].mul_(g["momentum"]).add_(p.grad)
                u = newton_schulz(p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf, g["ns_steps"])
                rows, cols = p.shape
                s = max(1.0, rows / cols) ** 0.5 if g["scale"] == "shape" else 0.2 * max(rows, cols) ** 0.5
                if g["wd"]:
                    p.mul_(1 - g["lr"] * g["wd"])
                p.add_(u, alpha=-g["lr"] * s)


def lr_mult(step: int, n: int, a: argparse.Namespace) -> float:
    warm = a.warmup * n if a.warmup < 1 else a.warmup
    ramp = min(1.0, (step + 1) / warm) if warm > 0 else 1.0
    t = step / n
    if a.sched == "linear":
        d = 1 - t
    elif a.sched == "cosine":
        d = 0.5 * (1 + math.cos(math.pi * t))
    elif a.sched == "const":
        d = 1.0
    else:
        start = 1 - a.decay_frac
        p = 0.0 if t < start else (t - start) / a.decay_frac
        d = 1 - math.sqrt(p) if a.decay_shape == "sqrt" else 1 - p
    return ramp * (a.final_frac + (1 - a.final_frac) * d)


def seen_windows(rec: dict, n_windows: int) -> np.ndarray:
    """The 1024-token windows the start read: its run's seed-0 order, the first step x batch (starts.py)."""
    p = rec["provenance"]
    if rec["config"]["family"] != "lf" or "step" not in p or "batch" not in p:
        return np.zeros(0, dtype=np.int64)
    return np.random.default_rng(p.get("data_seed", 0)).permutation(n_windows)[: int(p["step"]) * int(p["batch"])]


class Windows:
    """Rows of ctx + 1 tokens from TrainData's stream, drawn from a list of 1024-window ids in the given order (cycled when
    a run needs more): each id gives its 1024 // ctx consecutive sub-windows, in the same step."""

    def __init__(self, tokens: np.memmap, ids: np.ndarray):
        self.tokens, self.ids, self.at = tokens, ids, 0

    def take(self, n_ids: int) -> np.ndarray:
        pos = (self.at + np.arange(n_ids)) % len(self.ids)
        self.at = (self.at + n_ids) % len(self.ids)
        return self.ids[pos]

    def rows(self, ids: np.ndarray, ctx: int, length: int | None = None) -> torch.Tensor:
        """Each id's 1024 // ctx sub-windows of ctx + 1 tokens, or (`length`) only its first length + 1 tokens."""
        if length is not None:
            r = np.stack([self.tokens[j * CTX: j * CTX + length + 1] for j in ids])
        else:
            k = CTX // ctx
            sub = (ids[:, None] * k + np.arange(k)[None, :]).reshape(-1)
            r = np.stack([self.tokens[j * ctx: j * ctx + ctx + 1] for j in sub])
        return torch.from_numpy(r.astype(np.int64)).to(prepare.DEVICE, non_blocking=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mat-opt", choices=("muon", "adamw"), default="muon")
    ap.add_argument("--mat-lr", type=float, default=0.01, help="the block matrices' rate (Muon's or AdamW's)")
    ap.add_argument("--embed-lr", type=float, default=3e-4)
    ap.add_argument("--norm-lr", type=float, default=None, help="default: --embed-lr")
    ap.add_argument("--wd", type=float, default=0.0, help="decoupled weight decay on the block matrices")
    ap.add_argument("--embed-wd", type=float, default=0.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--momentum", type=float, default=0.95, help="Muon's")
    ap.add_argument("--no-nesterov", action="store_true")
    ap.add_argument("--muon-scale", choices=("shape", "rms"), default="shape")
    ap.add_argument("--ns-steps", type=int, default=5)
    ap.add_argument("--freeze-embed", action="store_true")
    ap.add_argument("--freeze-norms", action="store_true")
    ap.add_argument("--top-blocks", type=int, default=0,
                    help="train only the top K blocks (matrices and gains) and the final norm; the embedding and the "
                         "blocks below stay frozen, so the backward stops below block L-K (0: every block)")
    ap.add_argument("--block-lr-decay", type=float, default=1.0,
                    help="block i's matrices and gains train at D^(L-1-i) times their rate (the top block at 1)")
    ap.add_argument("--class", dest="klass", choices=sorted(prepare.CLASSES), default=KLASS,
                    help="the Delta bin the submission declares (the launch's CLASS)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--sched", choices=("linear", "wsd", "cosine", "const"), default="linear")
    ap.add_argument("--warmup", type=float, default=50, help="steps, or a fraction of the steps when below 1")
    ap.add_argument("--decay-frac", type=float, default=0.2, help="wsd: the decaying share of the steps")
    ap.add_argument("--decay-shape", choices=("linear", "sqrt"), default="linear")
    ap.add_argument("--final-frac", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--fresh", action="store_true", help="continue the start's own data order past what it trained on")
    ap.add_argument("--ema-frac", type=float, default=0.0)
    ap.add_argument("--val-rows", type=int, default=64, help="held-back training-stream windows per validation")
    ap.add_argument("--snap-at", default="0.25,0.5,0.75", help="fractions of the steps (comma-separated; empty: none)")
    ap.add_argument("--forks", default="", help="fractions of the steps where a cooldown branch forks (comma-separated)")
    ap.add_argument("--fork-len", type=float, default=0.15, help="a branch's steps, as a fraction of the steps before it")
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--windows", choices=("random", "fresh", "seen"), default="random")
    ap.add_argument("--short-ctx", type=int, default=CTX)
    ap.add_argument("--short-frac", type=float, default=0.0)
    ap.add_argument("--score-len", type=int, default=0)
    ap.add_argument("--keep", type=float, default=1.0)
    ap.add_argument("--select", choices=("drop-high", "drop-low", "mid"), default="drop-high")
    ap.add_argument("--order", choices=("random", "easy-first", "hard-first"), default="random")
    ap.add_argument("--trunc", type=float, default=0.0)
    a = ap.parse_args()
    a.norm_lr = a.embed_lr if a.norm_lr is None else a.norm_lr
    if CTX % a.short_ctx or not 0 <= a.short_frac <= 1:
        raise SystemExit("train: --short-ctx must divide 1024 and --short-frac lie in [0, 1]")
    if a.order != "random" and not a.score_len:
        raise SystemExit("train: --order needs --score-len (θ's scores)")
    if a.fresh and a.windows != "random":
        raise SystemExit("train: --fresh is ref-optim's fresh windows; give it or --windows, not both")
    if a.forks and a.score_len:
        raise SystemExit("train: a fork's windows are not scored; give --forks or --score-len, not both")
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = prepare.DEVICE.split(":")[0]
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    for p in params.values():
        p.data = p.data.float()
    embed = "embed" if rec["config"]["family"] == "lf" else "model.embed_tokens.weight"
    n_layer = rec["config"]["n_layer"]
    if (a.top_blocks or a.block_lr_decay != 1.0) and rec["config"]["family"] != "lf":
        raise SystemExit("train: --top-blocks and --block-lr-decay name lf blocks (blocks.N.*)")
    if not 0 <= a.top_blocks <= n_layer:
        raise SystemExit(f"train: --top-blocks must lie in [0, {n_layer}]")

    def block_of(n: str) -> int | None:
        m = re.match(r"blocks\.(\d+)\.", n)
        return int(m[1]) if m else None

    low = n_layer - a.top_blocks if a.top_blocks else 0
    mats = [n for n, p in params.items() if p.ndim == 2 and n != embed and (block_of(n) or 0) >= low]
    norms = [n for n, p in params.items() if p.ndim == 1 and (block_of(n) is None or block_of(n) >= low)]
    trained = mats + ([] if a.freeze_embed or a.top_blocks else [embed]) + ([] if a.freeze_norms else norms)
    for n, p in params.items():
        p.requires_grad_(n in trained)

    def by_rate(names: list[str], lr: float) -> list[dict]:
        """Param groups of `names`, block i's at lr * D^(L-1-i) (the final norm and the embedding at lr)."""
        rate = {}
        for n in names:
            b = block_of(n)
            rate.setdefault(lr * a.block_lr_decay ** (n_layer - 1 - b) if b is not None else lr, []).append(params[n])
        return [{"params": ps, "lr": r, "base_lr": r} for r, ps in rate.items()]

    opts = []
    adam_groups = []
    if a.mat_opt == "muon":
        opts.append(Muon(by_rate(mats, a.mat_lr), lr=a.mat_lr, momentum=a.momentum, nesterov=not a.no_nesterov,
                         wd=a.wd, scale=a.muon_scale, ns_steps=a.ns_steps))
    else:
        adam_groups += [g | {"weight_decay": a.wd} for g in by_rate(mats, a.mat_lr)]
    if embed in trained:
        adam_groups.append({"params": [params[embed]], "lr": a.embed_lr, "base_lr": a.embed_lr, "weight_decay": a.embed_wd})
    if not a.freeze_norms:
        adam_groups += [g | {"weight_decay": 0.0} for g in by_rate(norms, a.norm_lr)]
    if adam_groups:
        opts.append(torch.optim.AdamW(adam_groups, betas=(a.beta1, a.beta2), fused=dev == "cuda"))
    trainable = {"full": sorted(trained)}
    tokens = a.batch * CTX
    extra = sum(ns_flops(tuple(params[n].shape), a.ns_steps) for n in mats) if a.mat_opt == "muon" else 0

    def flops(op: str, ctx: int, n_tokens: int) -> int:
        return prepare.train_flops(rec, trainable, [{"op": op, "tokens": n_tokens, "ctx": ctx}])["flops"]

    step_long = flops("train", CTX, tokens) + extra
    step_short = flops("train", a.short_ctx, tokens) + extra
    per_step = a.short_frac * step_short + (1 - a.short_frac) * step_long
    snap_at = [float(x) for x in a.snap_at.split(",") if x.strip()]
    fork_at = [float(x) for x in a.forks.split(",") if x.strip()]
    val_tokens = a.val_rows * CTX
    n_val = ((len(snap_at) + 1) * (2 if a.ema_frac else 1) + len(fork_at)) if a.val_rows else 0
    val_flops = n_val * flops("forward", CTX, val_tokens)
    score_per_window = flops("forward", a.score_len, a.score_len) if a.score_len else 0
    pool_per_step = a.batch / a.keep if a.score_len else 0
    room = a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - val_flops
    main_step = per_step + pool_per_step * score_per_window

    def fork_lens(n: int) -> dict[int, int]:
        return {max(1, round(n * f)): max(1, round(a.fork_len * n * f)) for f in fork_at if 0 < f < 1}

    n_steps = int(room // (main_step + step_long * a.fork_len * sum(fork_at)))
    while n_steps > 1 and n_steps * main_step + sum(fork_lens(n_steps).values()) * step_long > room:
        n_steps -= 1
    forks = fork_lens(n_steps)
    n_short = round(a.short_frac * n_steps)

    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    n_windows = len(data.order)
    seen = seen_windows(rec, n_windows)
    if a.fresh:
        prov = rec["provenance"]
        if rec["config"]["family"] != "lf" or prov.get("data_seed") != 0:
            raise SystemExit("train: --fresh needs an lf start trained on data order 0")
        ids = np.random.default_rng(0).permutation(n_windows)[int(prov["step"]) * int(prov["batch"]):]
    else:
        ids = np.asarray(data.order)
    unseen = ~np.isin(ids, seen)
    val_idx = ids[np.flatnonzero(unseen)[len(np.flatnonzero(unseen)) - a.val_rows:]] if a.val_rows else ids[:0]
    keep_mask = ~np.isin(ids, val_idx)
    if a.windows == "fresh":
        keep_mask &= unseen
    elif a.windows == "seen":
        keep_mask &= ~unseen
    ids = ids[keep_mask]
    win = Windows(data.tokens, ids)

    def seq_loss(x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(dev, dtype=torch.bfloat16):
            logits = net(x[:, :-1])
        nll = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="none")
        return nll.view(x.shape[0], -1).mean(1)

    seqf = seq_loss if a.no_compile else torch.compile(seq_loss)
    steps: list[dict] = []

    @torch.no_grad()
    def validate() -> float:
        total = 0.0
        for i in range(0, a.val_rows, a.micro):
            idx = np.sort(val_idx[i: i + a.micro])
            total += float(seqf(win.rows(idx, CTX)).sum())
        steps.append({"op": "forward", "tokens": val_tokens, "what": "validation on held-back training windows"})
        return total / a.val_rows

    ema = [p.detach().clone() for p in params.values()] if a.ema_frac else None
    plist = list(params.values())

    def swap_ema() -> None:
        for p, e in zip(plist, ema):
            tmp = p.data.clone()
            p.data.copy_(e)
            e.copy_(tmp)

    def choose() -> tuple[str, dict]:
        """The weights to submit: raw, or EMA when its held-back loss is lower; logs both losses."""
        if not a.val_rows:
            return "raw", {}
        raw = validate()
        if ema is None:
            return "raw", {"val_raw": raw}
        swap_ema()
        ev = validate()
        if ev < raw:
            m = model_of(params, theta, trained)
            swap_ema()
            return "ema", {"val_raw": raw, "val_ema": ev, "model": m}
        swap_ema()
        return "raw", {"val_raw": raw, "val_ema": ev}

    info = {"steps": n_steps, "short_steps": n_short, "flops_per_step_long": step_long, "flops_per_step_short": step_short,
            "tokens_per_step": tokens, "val_flops": val_flops, "trained": len(trained), "stream_tokens": len(data.tokens),
            "windows_total": n_windows,
            "windows_seen_by_start": int(len(seen)), "windows_drawable": int(len(ids))}
    t0 = time.perf_counter()
    if a.score_len:
        pool = win.take(int(round(n_steps * pool_per_step)))
        sc = np.empty(len(pool), dtype=np.float32)
        rows_per = max(1, a.micro * CTX // a.score_len)
        # one step per chunk: the harness counts a step's tokens once per index buffer, and the chunks reuse one, so a
        # single scoring step would read out a score per window against one chunk's tokens, which the harness refuses
        for i in range(0, len(pool), rows_per):
            chunk = pool[i: i + rows_per]
            with hooks.step(), torch.no_grad():
                sc[i: i + len(chunk)] = seqf(win.rows(chunk, CTX, length=a.score_len)).float().cpu().numpy()
        steps.append({"op": "forward", "tokens": int(len(pool)) * a.score_len, "ctx": a.score_len,
                      "what": f"theta scores {len(pool)} windows on their first {a.score_len} tokens"})
        n_keep = n_steps * a.batch
        rank = np.argsort(sc, kind="stable")
        drop = len(pool) - n_keep
        lo = {"drop-high": 0, "drop-low": drop, "mid": drop // 2}[a.select]
        kept = np.sort(rank[lo: lo + n_keep])
        if a.order == "easy-first":
            kept = kept[np.argsort(sc[kept], kind="stable")]
        elif a.order == "hard-first":
            kept = kept[np.argsort(-sc[kept], kind="stable")]
        else:
            kept = np.random.default_rng(a.data_seed + 3).permutation(kept)
        plan = pool[kept]
        sixth = np.minimum(pool * 6 // n_windows, 5)
        info |= {"pool": int(len(pool)), "kept": int(n_keep), "pool_loss": float(sc.mean()),
                 "kept_loss": float(sc[kept].mean()),
                 "pool_loss_q10_50_90": [float(v) for v in np.quantile(sc, [0.1, 0.5, 0.9])],
                 "pool_loss_by_stream_sixth": [float(sc[sixth == k].mean()) if (sixth == k).any() else None for k in range(6)],
                 "score_seconds": round(time.perf_counter() - t0, 1)}
    else:
        plan = win.take(n_steps * a.batch)
    fork_plan = win.take(sum(forks.values()) * a.batch)
    info["plan_seen_share"] = float(np.isin(plan, seen).mean()) if len(seen) else 0.0
    info["plan_repeats"] = int(len(plan) - len(np.unique(plan)))
    info["forks"] = forks

    marks = {max(1, round(n_steps * f)) for f in snap_at if 0 < f < 1}
    out = Path(a.out)
    snap_log = []
    print(json.dumps(info), flush=True)
    trained_params = [params[n] for n in trained]

    def train_step(m: float, ids: np.ndarray, ctx: int) -> float:
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * m
        rows = win.rows(ids, ctx)
        micro = a.micro * (CTX // ctx)
        chunks = [rows[i: i + micro] for i in range(0, len(rows), micro)]
        n_kept = sum(len(x) - int(a.trunc * len(x)) for x in chunks)
        total = 0.0
        for x in chunks:
            seq = seqf(x)
            total += float(seq.detach().sum())
            n_drop = int(a.trunc * len(x))
            if n_drop:
                keep = torch.ones_like(seq)
                keep[torch.topk(seq.detach(), n_drop).indices] = 0.0
                seq = seq * keep
            (seq.sum() / n_kept).backward()
        if a.clip:
            torch.nn.utils.clip_grad_norm_(trained_params, a.clip)
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)
        steps.append({"op": "train", "tokens": tokens, "ctx": ctx})
        if extra:
            steps.append({"op": "extra", "flops": extra, "what": "muon newton-schulz"})
        return total / len(rows)

    def snapshot(model: dict, entry: dict) -> None:
        d = out / "snap" / f"{len(snap_log):03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "E.bin").write_bytes(prepare.encode_full(model))
        (d / "meta.json").write_text(json.dumps({"steps_prefix": len(steps)}) + "\n")
        snap_log.append(entry)
        print(json.dumps({"snapshot": len(snap_log) - 1, **entry}), flush=True)

    fork_at_ids = iter(range(0, len(fork_plan), a.batch))

    @torch.no_grad()
    def fork(step: int, k: int) -> None:
        """ref-optim's cooldown branch: k steps from the current rate linearly toward 0 (at 1024, on windows after the plan),
        snapshot, then back to the weights and optimizer state it forked from."""
        saved = [p.detach().clone() for p in trained_params]
        states = [{p: {n: (v.clone() if torch.is_tensor(v) else v) for n, v in st.items()} for p, st in o.state.items()}
                  for o in opts]
        m0 = lr_mult(step - 1, n_steps, a)
        with torch.enable_grad():
            for j in range(k):
                i = next(fork_at_ids)
                train_step(m0 * (k - j) / k, fork_plan[i: i + a.batch], CTX)
        entry = {"step": step, "fork_steps": k, "weights": "fork"}
        if a.val_rows:
            entry["val_raw"] = validate()
        snapshot(model_of(params, theta, trained), entry)
        for p, s in zip(trained_params, saved):
            p.copy_(s)
        for o, st in zip(opts, states):
            o.state.clear()
            o.state.update(st)

    for step in range(n_steps):
        with hooks.step():
            m = lr_mult(step, n_steps, a)
            ctx = a.short_ctx if step < n_short else CTX
            loss = train_step(m, plan[step * a.batch: (step + 1) * a.batch], ctx)
            if ema is not None:
                beta = 1 - 1 / max(1.0, a.ema_frac * (step + 1))
                torch._foreach_lerp_(ema, [p.detach() for p in plist], 1 - beta)
            if (step + 1) % 100 == 0:
                print(json.dumps({"step": step + 1, "ctx": ctx, "loss": loss, "lr_mult": round(m, 4),
                                  "s": round(time.perf_counter() - t0, 1)}), flush=True)
            if step + 1 in marks:
                which, vinfo = choose()
                model = vinfo.pop("model", None) or model_of(params, theta, trained)
                snapshot(model, {"step": step + 1, "weights": which, **vinfo})
            if step + 1 in forks:
                fork(step + 1, forks[step + 1])
    with hooks.step():
        which, vinfo = choose()
        model = vinfo.pop("model", None) or model_of(params, theta, trained)
        final = {"step": n_steps, "weights": which, **vinfo}
        print(json.dumps({"final": final, "s": round(time.perf_counter() - t0, 1)}), flush=True)
        prepare.submit(out, a.start, a.klass, prepare.encode_full(model), {**theta, **model}, trainable, steps,
                       changed=sorted(trained), notes={"direction": "ref-data", "args": vars(a), "steps": n_steps,
                                                       "data": info, "snapshots": snap_log, "final": final})
    return 0


def model_of(params: dict, theta: dict, trained: list[str]) -> dict:
    return {n: params[n].detach().to(theta[n].dtype).cpu() for n in trained}


if __name__ == "__main__":
    raise SystemExit(main())
