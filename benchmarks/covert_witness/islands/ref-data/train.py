"""Island `ref-data` (class `reference`, no cap on the update): the seed's Muon + AdamW continuation of every tensor, with the
data it trains on and their order as the variable. Every FLOP spent choosing data is in the step log.

The launcher runs `islands/reference/train.py`, a symlink to this file. Data flags (all off by default, which is the seed):

- `--windows random|fresh|seen`: the 1024-token windows drawn from TrainData's stream. `fresh` drops the windows the start
  read (its provenance's step x batch first windows of the seed-0 order, `starts.py`), `seen` keeps only those.
- `--short-ctx C --short-frac F`: the first F of the steps at context C (C divides 1024; the same tokens per step), priced
  at that context; the rest at 1024.
- `--score-len P --keep K --select drop-high|drop-low|mid`: θ scores a pool of windows by its mean loss on each window's
  first P tokens (a `forward` pass at context P, priced) and K of the pool is kept.
- `--order random|easy-first|hard-first`: the kept windows' order by θ's score (needs `--score-len`).
- `--trunc Q`: in each micro-batch the Q of the sequences with the highest loss get no gradient (loss truncation; no
  extra FLOPs, their pass is still priced).

    python guard.py islands/reference/train.py --start s100 --out SUB [--lr-scale 1] [--budget-fraction 0.98] [data flags]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

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
    def __init__(self, params, lr: float, momentum: float = 0.95, wd: float = 0.0):
        super().__init__(params, {"lr": lr, "momentum": momentum, "wd": wd})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state.setdefault(p, {"buf": torch.zeros_like(p)})
                buf = st["buf"].mul_(g["momentum"]).add_(p.grad)
                u = newton_schulz(p.grad.add(buf, alpha=g["momentum"]))
                if g["wd"]:
                    p.mul_(1 - g["lr"] * g["wd"])
                p.add_(u, alpha=-g["lr"] * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)


def seen_windows(rec: dict, n_windows: int) -> np.ndarray:
    """The 1024-token windows the start read: its run's seed-0 order, the first step x batch (starts.py)."""
    p = rec["provenance"]
    if "step" not in p or "batch" not in p:
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

    def rows(self, ids: np.ndarray, ctx: int, device: str) -> torch.Tensor:
        k = CTX // ctx
        sub = (ids[:, None] * k + np.arange(k)[None, :]).reshape(-1)
        last = (len(self.tokens) - 1) // ctx
        sub = sub[sub < last]
        rows = np.stack([self.tokens[j * ctx: j * ctx + ctx + 1] for j in sub]).astype(np.int64)
        return torch.from_numpy(rows).to(device, non_blocking=True)


@torch.no_grad()
def score_windows(fwd, win: Windows, ids: np.ndarray, plen: int, micro: int) -> np.ndarray:
    """θ's mean next-token loss on each window's first `plen` tokens."""
    out = np.empty(len(ids), dtype=np.float32)
    rows_per = max(1, micro * CTX // plen)
    for i in range(0, len(ids), rows_per):
        chunk = ids[i: i + rows_per]
        x = torch.from_numpy(np.stack([win.tokens[j * CTX: j * CTX + plen + 1] for j in chunk]).astype(np.int64))
        x = x.to(prepare.DEVICE, non_blocking=True)
        with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
            logits = fwd(x[:, :-1])
        nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="none")
        out[i: i + len(chunk)] = nll.view(len(chunk), -1).mean(1).cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--muon-lr", type=float, default=0.01)
    ap.add_argument("--adam-lr", type=float, default=3e-4)
    ap.add_argument("--lr-scale", type=float, default=1.0, help="multiplies both learning rates (the screens' sweep)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    ap.add_argument("--snapshots", type=int, default=3)
    ap.add_argument("--windows", choices=("random", "fresh", "seen"), default="random")
    ap.add_argument("--short-ctx", type=int, default=CTX)
    ap.add_argument("--short-frac", type=float, default=0.0)
    ap.add_argument("--score-len", type=int, default=0)
    ap.add_argument("--keep", type=float, default=1.0)
    ap.add_argument("--select", choices=("drop-high", "drop-low", "mid"), default="drop-high")
    ap.add_argument("--order", choices=("random", "easy-first", "hard-first"), default="random")
    ap.add_argument("--trunc", type=float, default=0.0)
    a = ap.parse_args()
    if CTX % a.short_ctx or not 0 <= a.short_frac <= 1:
        raise SystemExit("train: --short-ctx must divide 1024 and --short-frac lie in [0, 1]")
    if a.order != "random" and not a.score_len:
        raise SystemExit("train: --order needs --score-len (θ's scores)")
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    for p in params.values():
        p.data = p.data.float()
    embed = "embed" if rec["config"]["family"] == "lf" else "model.embed_tokens.weight"
    mats = [n for n, p in params.items() if p.ndim == 2 and n != embed]
    rest = [n for n in params if n not in mats]
    muon_lr, adam_lr = a.muon_lr * a.lr_scale, a.adam_lr * a.lr_scale
    muon = Muon([params[n] for n in mats], lr=muon_lr)
    adam = torch.optim.AdamW([params[n] for n in rest], lr=adam_lr, betas=(0.9, 0.95), weight_decay=0.0,
                             fused=prepare.DEVICE.startswith("cuda"))
    trainable = {"full": sorted(params)}
    tokens = a.batch * CTX
    extra = sum(ns_flops(tuple(params[n].shape)) for n in mats)

    def flops(op: str, ctx: int, n_tokens: int) -> int:
        return prepare.train_flops(rec, trainable, [{"op": op, "tokens": n_tokens, "ctx": ctx}])["flops"]

    step_long = flops("train", CTX, tokens) + extra
    step_short = flops("train", a.short_ctx, tokens) + extra
    per_step = a.short_frac * step_short + (1 - a.short_frac) * step_long
    available = a.budget_fraction * prepare.TRAIN_FLOP_BUDGET
    score_per_window = flops("forward", a.score_len, a.score_len) if a.score_len else 0
    pool_per_step = a.batch / a.keep if a.score_len else 0
    n_steps = int(available // (per_step + pool_per_step * score_per_window))
    n_short = round(a.short_frac * n_steps)

    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    n_windows = len(data.order)
    seen = seen_windows(rec, n_windows)
    ids = np.random.default_rng(a.data_seed).permutation(n_windows)
    if a.windows != "random":
        mask = np.isin(ids, seen)
        ids = ids[mask] if a.windows == "seen" else ids[~mask]
    win = Windows(data.tokens, ids)
    fwd = torch.compile(net)
    steps, snaps = [], []
    info = {"steps": n_steps, "short_steps": n_short, "flops_per_step_long": step_long, "flops_per_step_short": step_short,
            "tokens_per_step": tokens, "windows_total": n_windows, "windows_seen_by_start": int(len(seen)),
            "windows_drawable": int(len(ids))}
    t0 = time.perf_counter()
    plan = win.take(n_steps * a.batch) if not a.score_len else None
    if a.score_len:
        pool = win.take(int(round(n_steps * pool_per_step)))
        net.eval()
        sc = score_windows(fwd, win, pool, a.score_len, a.micro)
        net.train()
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
        q = np.quantile(sc, [0.1, 0.5, 0.9])
        info |= {"pool": int(len(pool)), "kept": int(n_keep), "pool_loss": float(sc.mean()),
                 "kept_loss": float(sc[kept].mean()), "pool_loss_q10_50_90": [float(v) for v in q],
                 "score_seconds": round(time.perf_counter() - t0, 1)}
    info["plan_seen_share"] = float(np.isin(plan, seen).mean()) if len(seen) else 0.0
    print(json.dumps(info), flush=True)
    marks = {round(n_steps * k / (a.snapshots + 1)) for k in range(1, a.snapshots + 1)}
    for step in range(n_steps):
        frac = min(1.0, (step + 1) / a.warmup) * (1 - step / n_steps)
        for o, lr in ((muon, muon_lr), (adam, adam_lr)):
            for g in o.param_groups:
                g["lr"] = lr * frac
        ctx = a.short_ctx if step < n_short else CTX
        rows = win.rows(plan[step * a.batch: (step + 1) * a.batch], ctx, prepare.DEVICE)
        micro = a.micro * (CTX // ctx)
        n_rows = len(rows)
        chunks = [rows[i: i + micro] for i in range(0, n_rows, micro)]
        n_kept = sum(len(x) - int(a.trunc * len(x)) for x in chunks)
        for x in chunks:
            with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            nll = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="none")
            raw = nll.view(len(x), -1).mean(1)
            seq = raw
            n_drop = int(a.trunc * len(x))
            if n_drop:
                keep = torch.ones_like(seq)
                keep[torch.topk(seq.detach(), n_drop).indices] = 0.0
                seq = seq * keep
            (seq.sum() / n_kept).backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
        muon.step()
        adam.step()
        muon.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)
        steps += [{"op": "train", "tokens": tokens, "ctx": ctx}, {"op": "extra", "flops": extra, "what": "muon newton-schulz"}]
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "ctx": ctx, "loss": float(raw.detach().mean()),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if step + 1 in marks:
            snaps.append((prepare.encode_full(model_of(params, theta)), len(steps)))
    model = model_of(params, theta)
    prepare.submit(prepare.Path(a.out), a.start, KLASS, prepare.encode_full(model), model, trainable, steps,
                   changed=sorted(params), notes={"optimizer": "muon+adamw", "args": vars(a), "data": info},
                   snapshots=snaps)
    return 0


def model_of(params: dict, theta: dict) -> dict:
    return {n: p.detach().to(theta[n].dtype).cpu() for n, p in params.items()}


if __name__ == "__main__":
    raise SystemExit(main())
