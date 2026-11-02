"""Direction `hybrid` (any class): one script that composes the other directions' mechanisms, so a pair is a flag change.

- `--full SPEC`: tensors trained in full, by AdamW or Muon (`--opt`); `--lora SPEC`: rank-r adapters (`--rank`, 0 the largest
  that fits the class's cap), AdamW with LoRA+'s faster B (`--lora-b-mult`); `--norms`: norm gains above the lowest updated
  matrix (0-FLOP nodes, so φ is unchanged) or none; `--final-norm`.
- `--store bf16` writes every full tensor through the `lora` format's bf16 `full` list: the evaluator multiplies weights at
  bf16 under autocast, so the f32 low halves `encode_full` would carry never reach a matmul. `--store f32` is `encode_full`.
- `--select-pool P`: draw P·batch windows, score them with a forward (logged), train on the `--select-by` share.

SPEC is comma-separated `BLOCKS:LEAVES`: BLOCKS `all`, `i`, `i-j` or a negative index from the top; LEAVES `+`-joined leaf
names (`wq wk wv wo w_gate w_up w_down`) or groups (`qkv`, `attn`, `gu`, `mlp`, `all`). `--klass` must equal the class the
launcher runs (on this branch `islands/<class>/train.py` links here).

    python guard.py islands/hybrid/train.py --klass shallow --start s100 --out SUB --full -1:gu --final-norm --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare

LEAVES = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")
GROUPS = {"qkv": ("wq", "wk", "wv"), "attn": ("wq", "wk", "wv", "wo"), "gu": ("w_gate", "w_up"),
          "mlp": ("w_gate", "w_up", "w_down"), "all": LEAVES}


def select(spec: str, L: int) -> list[str]:
    out = set()
    for item in filter(None, spec.split(",")):
        blocks, leaves = item.split(":")
        if blocks == "all":
            idx = range(L)
        elif re.fullmatch(r"\d+-\d+", blocks):
            lo, hi = map(int, blocks.split("-"))
            idx = range(lo, hi + 1)
        else:
            idx = [int(blocks) % L]
        names = [x for leaf in leaves.split("+") for x in GROUPS.get(leaf, (leaf,))]
        bad = set(names) - set(LEAVES)
        if bad:
            raise SystemExit(f"hybrid: unknown leaves {sorted(bad)}")
        out |= {f"blocks.{i}.{n}.weight" for i in idx for n in names}
    return sorted(out)


def pos(rec: dict) -> dict[str, tuple[int, int]]:
    return {n["param"]: (n["layer"], n["stage"]) for n in prepare.positions(rec) if n["flops"] or n["param"] != "embed"}


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


def encode(store: str, factors: dict, params: dict, full: list[str], theta: dict) -> bytes:
    if store == "f32":
        if factors:
            raise SystemExit("hybrid: --store f32 is encode_full, which holds no adapters")
        return prepare.encode_full({n: params[n].detach().to(theta[n].dtype) for n in full})
    return prepare.encode_lora({n: (A, B, s) for n, (A, B, s) in factors.items()}, "bf16", {n: params[n] for n in full})


def e_bytes(store: str, theta: dict, lora: list[str], rank: int, full: list[str]) -> int:
    fac = {n: (torch.zeros(rank, theta[n].shape[1]), torch.zeros(theta[n].shape[0], rank), math.sqrt(theta[n].shape[1] / rank))
           for n in lora}
    return len(encode(store, fac, theta, full, theta))


def lr_frac(step: int, n: int, warmup: int, schedule: str, cooldown: float) -> float:
    w = min(1.0, (step + 1) / warmup)
    if schedule == "linear":
        return w * (1 - step / n)
    if schedule == "cosine":
        return w * 0.5 * (1 + math.cos(math.pi * step / n))
    tail = max(1, round(cooldown * n))
    return w * min(1.0, (n - step) / tail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--klass", required=True, choices=sorted(prepare.CLASSES))
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", default="", help="SPEC of matrices trained in full")
    ap.add_argument("--lora", default="", help="SPEC of matrices trained through adapters")
    ap.add_argument("--rank", type=int, default=0, help="0: the largest that fits the class's cap")
    ap.add_argument("--norms", choices=("none", "above"), default="above")
    ap.add_argument("--final-norm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--store", choices=("bf16", "f32"), default="bf16")
    ap.add_argument("--opt", choices=("adamw", "muon"), default="adamw", help="for the full matrices")
    ap.add_argument("--lr", type=float, default=3e-4, help="full matrices (AdamW) or Muon's rate")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--norm-lr", type=float, default=3e-4)
    ap.add_argument("--lora-lr", type=float, default=1e-3)
    ap.add_argument("--lora-b-mult", type=float, default=1.0)
    ap.add_argument("--schedule", choices=("linear", "cosine", "wsd"), default="linear")
    ap.add_argument("--cooldown", type=float, default=0.3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--select-pool", type=int, default=1, help="windows drawn per trained window")
    ap.add_argument("--select-by", choices=("high", "low", "mid"), default="high")
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
    if rec["config"]["family"] != "lf":
        raise SystemExit("hybrid: lf starts only")
    params = net.params()
    L = rec["config"]["n_layer"]
    full_m, lora = select(a.full, L), select(a.lora, L)
    if set(full_m) & set(lora):
        raise SystemExit(f"hybrid: in both --full and --lora: {sorted(set(full_m) & set(lora))[:3]}")
    if not full_m and not lora:
        raise SystemExit("hybrid: nothing to train")
    where = pos(rec)
    lowest = min(where[n] for n in full_m + lora)
    gains = [n for n in params if params[n].ndim == 1 and n != "norm"]
    norms = sorted(n for n in gains if a.norms == "above" and where[n] > lowest) + (["norm"] if a.final_norm else [])
    full = sorted(full_m + norms)
    cap = prepare.CLASSES[a.klass]["max_e_bytes"]
    rank = a.rank
    if lora and not rank:
        if cap is None:
            raise SystemExit("hybrid: --rank 0 needs a class with a byte cap")
        rank = 1
        while e_bytes(a.store, theta, lora, rank + 1, full) <= cap - 512:
            rank += 1
    size = e_bytes(a.store, theta, lora, rank, full)
    if cap is not None and size > cap:
        raise SystemExit(f"hybrid: E is {size} bytes, over {a.klass}'s cap of {cap}")
    for p in params.values():
        p.requires_grad_(False)
    for n in full:
        params[n].data = params[n].data.float()
        params[n].requires_grad_(True)
    g = torch.Generator(device=prepare.DEVICE).manual_seed(a.seed)
    factors = {}
    for n in lora:
        w = params[n]
        A = torch.nn.Parameter(torch.randn(rank, w.shape[1], device=prepare.DEVICE, generator=g) / math.sqrt(w.shape[1]))
        B = torch.nn.Parameter(torch.zeros(w.shape[0], rank, device=prepare.DEVICE))
        s = math.sqrt(w.shape[1] / rank)
        factors[n] = (A, B, s)
        mod = net.inner.get_submodule(n.removesuffix(".weight"))
        mod.register_forward_hook(lambda m, inp, out, A=A, B=B, s=s: out + s * F.linear(F.linear(inp[0], A), B))
    fused = prepare.DEVICE.startswith("cuda")
    groups = []
    if norms:
        groups.append({"params": [params[n] for n in norms], "lr": a.norm_lr, "weight_decay": 0.0})
    if factors:
        groups.append({"params": [A for A, _, _ in factors.values()], "lr": a.lora_lr, "weight_decay": 0.0})
        groups.append({"params": [B for _, B, _ in factors.values()], "lr": a.lora_lr * a.lora_b_mult, "weight_decay": 0.0})
    if full_m and a.opt == "adamw":
        groups.append({"params": [params[n] for n in full_m], "lr": a.lr, "weight_decay": a.wd})
    opts = [torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=fused)] if groups else []
    if full_m and a.opt == "muon":
        opts.append(Muon([params[n] for n in full_m], lr=a.lr, wd=a.wd))
    base_lrs = [[grp["lr"] for grp in o.param_groups] for o in opts]
    trainable = {"full": full, "adapters": {n: rank for n in lora}}
    tokens = a.batch * prepare.EVAL_CTX
    extra = sum(ns_flops(tuple(params[n].shape)) for n in full_m) if a.opt == "muon" else 0
    pool = a.select_pool * a.batch
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}]
                                   + ([{"op": "forward", "tokens": pool * prepare.EVAL_CTX}] if a.select_pool > 1 else []))["flops"] + extra
    n_steps = int(a.budget_fraction * prepare.TRAIN_FLOP_BUDGET // per_step)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    fwd = torch.compile(net) if fused else net
    dev = prepare.DEVICE.split(":")[0]
    structure = prepare.deep_share(rec, set(full + lora))
    print(json.dumps({"steps": n_steps, "full": len(full_m), "lora": len(lora), "rank": rank if lora else None,
                      "norms": len(norms), "E_bytes": size, "flops_per_step": per_step, "phi": structure["phi"],
                      "lowest_updated_layer": structure["lowest_updated_layer"]}), flush=True)
    steps, snaps = [], []
    marks = {round(n_steps * k / 4) for k in (1, 2, 3)}
    t0 = time.perf_counter()
    for step in range(n_steps):
        frac = lr_frac(step, n_steps, a.warmup, a.schedule, a.cooldown)
        for o, lrs in zip(opts, base_lrs):
            for grp, lr in zip(o.param_groups, lrs):
                grp["lr"] = lr * frac
        rows = data.next(pool)
        if a.select_pool > 1:
            rows = choose(fwd, rows, a.batch, a.micro, a.select_by, dev)
            steps.append({"op": "forward", "tokens": pool * prepare.EVAL_CTX, "what": f"selection by {a.select_by} loss"})
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            with torch.autocast(dev, dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / a.batch)
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for o in opts for grp in o.param_groups for p in grp["params"]], 1.0)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        steps.append({"op": "train", "tokens": tokens})
        if extra:
            steps.append({"op": "extra", "flops": extra, "what": "muon newton-schulz"})
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if step + 1 in marks:
            snaps.append((encode(a.store, factors, params, full, theta), len(steps)))
    blob = encode(a.store, factors, params, full, theta)
    model = prepare.decode(theta, blob)
    prepare.submit(Path(a.out), a.start, a.klass, blob, model, trainable, steps, changed=full + lora,
                   notes={"method": "hybrid", "rank": rank if lora else None, "args": vars(a), "steps": n_steps},
                   snapshots=snaps)
    return 0


@torch.no_grad()
def choose(fwd, rows: torch.Tensor, keep: int, micro: int, by: str, dev: str) -> torch.Tensor:
    losses = []
    for i in range(0, len(rows), micro):
        x = rows[i: i + micro]
        with torch.autocast(dev, dtype=torch.bfloat16):
            logits = fwd(x[:, :-1])
        losses.append(F.cross_entropy(logits.float().transpose(1, 2), x[:, 1:], reduction="none").mean(1))
    order = torch.cat(losses).argsort(descending=True)
    lo = {"high": 0, "mid": (len(rows) - keep) // 2, "low": len(rows) - keep}[by]
    return rows[order[lo: lo + keep]]


if __name__ == "__main__":
    raise SystemExit(main())
