"""Island `shallow` (only the top 2 blocks and the final norm may differ, any size): updates near the output, so few updated
multiplications lie upstream (low depth and deep share). The rest stays frozen and the backward stops at the lowest tuned
node. The head is the tied embedding, which the input lookup also reads (layer 0), so it stays frozen.

What is tuned: tensors trained in full (`--full`) and matrices trained through rank-`--lora` factors (`--adapt`), each a
comma list of `tN.LEAF` (block N from the top, t0 the last; LEAF one of LF_LEAVES or `*`) and `final` (the final norm).
Without them, `--blocks`/`--parts`/`--final` name the full set. The first updated stage has no updated multiplication
upstream, so it is never deep: `--full t0.w_gate,t0.w_up,final` has φ = 0 and depth 0, and `--full t1.w_gate,t1.w_up
--adapt t1.w_down,t0.* --lora 4` keeps every later update low-rank.

How: AdamW or Muon (`--opt`) on full matrices, AdamW on gains and factors. E (`--store`): `fp32` every changed tensor
in θ's dtype, adapters merged (`prepare.encode_full`); `bf16` factors and full tensors in bf16 (the lora format; the
bf16-autocast forward already multiplies at bf16); `svd:R` each full matrix's delta truncated to rank R as bf16 factors;
`auto` is bf16 with adapters, else fp32. `--fresh` trains only on windows the start never read (its provenance names its
order: default_rng(data_seed), step x batch). `--ema D` submits an exponential average of the full tensors. The submitted
model is always decode(θ, E).

This file is the island's one editable file; prepare.py, evaluate.py, guard.py and candidate.sh are fixed (README.md, the program).

    python guard.py islands/shallow/train.py --start m100 --out SUB [--full SPEC] [--adapt SPEC --lora R] [--opt adamw]
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

import prepare

KLASS = "shallow"
LF_LEAVES = ("attn_norm", "wq", "wk", "wv", "wo", "mlp_norm", "w_gate", "w_up", "w_down")
LF_PARAM = {leaf: leaf if leaf.endswith("norm") else f"{leaf}.weight" for leaf in LF_LEAVES}


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


def ns_flops(shape, steps: int = 5) -> int:
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


def spec_names(rec: dict, params: dict, spec: str) -> list[str]:
    L = rec["config"]["n_layer"]
    out = []
    for item in filter(None, spec.split(",")):
        if item == "final":
            out.append("norm")
            continue
        m = re.fullmatch(r"t(\d+)\.(\*|\w+)", item)
        if not m or (m.group(2) != "*" and m.group(2) not in LF_LEAVES) or int(m.group(1)) >= L:
            raise SystemExit(f"shallow: {item!r} is not tN.LEAF (LEAF in {LF_LEAVES} or *) or final")
        block = L - 1 - int(m.group(1))
        leaves = LF_LEAVES if m.group(2) == "*" else (m.group(2),)
        out += [f"blocks.{block}.{LF_PARAM[leaf]}" for leaf in leaves]
    missing = set(out) - set(params)
    if missing:
        raise SystemExit(f"shallow: no tensors {sorted(missing)[:4]}")
    return sorted(set(out))


def default_full(blocks: int, parts: str, final: bool) -> str:
    leaves = LF_LEAVES if parts == "all" else parts.split(",")
    return ",".join([f"t{t}.{leaf}" for t in range(blocks) for leaf in leaves] + (["final"] if final else []))


def fresh_order(data, rec: dict) -> dict:
    """Drop the windows the start read from the stream's order (its provenance: default_rng(data_seed), step x batch)."""
    prov = rec.get("provenance", {})
    if not {"step", "batch", "data_seed"} <= set(prov):
        return {"fresh": False, "why": "the start's provenance names no data order"}
    n_seq = (len(data.tokens) - 1) // data.ctx
    seen = np.random.default_rng(prov["data_seed"]).permutation(n_seq)[: prov["step"] * prov["batch"]]
    keep = np.ones(n_seq, dtype=bool)
    keep[seen] = False
    before = len(data.order)
    data.order = data.order[keep[data.order]]
    return {"fresh": True, "windows": n_seq, "seen": int(len(seen)), "kept": int(len(data.order)), "of": before}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", default=None, help="tensors trained in full: tN.LEAF, tN.*, final (default: --blocks/--parts)")
    ap.add_argument("--adapt", default="", help="matrices trained through rank --lora factors: tN.LEAF, tN.*")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--parts", default="all", help="`all` or leaf names in each top block: " + ",".join(LF_LEAVES))
    ap.add_argument("--final", action=argparse.BooleanOptionalAction, default=True, help="tune the final norm")
    ap.add_argument("--opt", choices=("adamw", "muon"), default="adamw")
    ap.add_argument("--lr", type=float, default=3e-4, help="AdamW's rate (gains, and full matrices unless --opt muon)")
    ap.add_argument("--muon-lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lora", type=int, default=0, help="rank of the --adapt factors (alone: every tuned matrix)")
    ap.add_argument("--lora-lr", type=float, default=1e-3)
    ap.add_argument("--store", default="auto", help="auto | fp32 | bf16 | svd:R")
    ap.add_argument("--fresh", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--schedule", choices=("linear", "cosine", "wsd"), default="linear")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snaps", type=int, default=10, help="snapshots at every 1/N of the run, for the FLOPs-to-target curve")
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    if prepare.start_record(a.start)["config"]["family"] != "lf":
        raise SystemExit("shallow: this train.py tunes the lf starts (s50, s100, m50, m100)")
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    full = spec_names(rec, params, a.full if a.full is not None else default_full(a.blocks, a.parts, a.final))
    adapt = [n for n in spec_names(rec, params, a.adapt) if params[n].ndim == 2]
    if a.lora and not a.adapt:
        adapt, full = [n for n in full if params[n].ndim == 2], [n for n in full if params[n].ndim == 1]
    if adapt and not a.lora:
        raise SystemExit("shallow: --adapt needs --lora R")
    if set(full) & set(adapt) or any(params[n].ndim != 2 for n in adapt):
        raise SystemExit("shallow: --adapt names matrices that --full does not")
    mats = [n for n in full if params[n].ndim == 2]
    gains = [n for n in full if params[n].ndim == 1]
    store = a.store if a.store != "auto" else ("bf16" if adapt else "fp32")
    for p in params.values():
        p.requires_grad_(False)
    for n in full:
        params[n].data = params[n].data.float()
        params[n].requires_grad_(True)

    factors = {}
    g = torch.Generator(device=prepare.DEVICE).manual_seed(a.seed)
    for n in adapt:
        w = params[n]
        A = torch.nn.Parameter(torch.randn(a.lora, w.shape[1], device=prepare.DEVICE, generator=g) / math.sqrt(w.shape[1]))
        B = torch.nn.Parameter(torch.zeros(w.shape[0], a.lora, device=prepare.DEVICE))
        s = math.sqrt(w.shape[1] / a.lora)
        factors[n] = (A, B, s)
        mod = net.inner.get_submodule(n.removesuffix(".weight"))
        mod.register_forward_hook(lambda m, inp, out, A=A, B=B, s=s: out + s * F.linear(F.linear(inp[0], A), B))

    fused = prepare.DEVICE.startswith("cuda")
    opts, extra = [], 0
    if factors:
        opts.append((torch.optim.AdamW([p for A, B, _ in factors.values() for p in (A, B)], lr=a.lora_lr, betas=(0.9, 0.95),
                                       weight_decay=0.0, fused=fused), a.lora_lr))
    if mats and a.opt == "muon":
        opts.append((Muon([params[n] for n in mats], lr=a.muon_lr, wd=a.wd), a.muon_lr))
        extra = sum(ns_flops(tuple(params[n].shape)) for n in mats)
    elif mats:
        opts.append((torch.optim.AdamW([params[n] for n in mats], lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd,
                                       fused=fused), a.lr))
    if gains:
        opts.append((torch.optim.AdamW([params[n] for n in gains], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0,
                                       fused=fused), a.lr))
    trained = [p for o, _ in opts for grp in o.param_groups for p in grp["params"]]

    trainable = {"full": full, "adapters": {n: a.lora for n in adapt}}
    tokens = a.batch * prepare.EVAL_CTX
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"] + extra
    store_extra = svd_flops(params, mats) if store.startswith("svd:") else 0
    n_steps = int((a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - (a.snaps + 1) * store_extra) // per_step)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    fresh = fresh_order(data, rec) if a.fresh else {"fresh": False}
    fwd = torch.compile(net) if a.compile else net
    ema = {n: params[n].detach().float().clone() for n in full} if a.ema else None
    steps, snaps = [], []
    marks = {round(n_steps * k / a.snaps) for k in range(1, a.snaps)} if a.snaps > 1 else set()
    info = {"steps": n_steps, "full": full, "adapt": adapt, "rank": a.lora, "store": store, "flops_per_step": per_step,
            "extra_per_step": extra, "fresh": fresh}
    print(json.dumps(info), flush=True)

    def encode() -> bytes:
        cur = {n: (ema[n] if ema is not None else params[n].detach()).float() for n in full}
        return encode_update(store, theta, cur, factors, mats, gains, params, steps)

    t0 = time.perf_counter()
    for step in range(n_steps):
        frac = min(1.0, (step + 1) / a.warmup) * decay(a.schedule, step, n_steps)
        for o, lr in opts:
            for grp in o.param_groups:
                grp["lr"] = lr * frac
        rows = data.next(a.batch)
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / a.batch)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(trained, 1.0)
        for o, _ in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        if ema is not None:
            with torch.no_grad():
                for n in full:
                    ema[n].lerp_(params[n].detach().float(), 1 - a.ema)
        steps.append({"op": "train", "tokens": tokens})
        if extra:
            steps.append({"op": "extra", "flops": extra, "what": "muon newton-schulz"})
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if step + 1 in marks:
            snaps.append((encode(), len(steps)))
    blob = encode()
    model = prepare.decode(theta, blob)
    tuned = sorted(full + adapt)
    changed = sorted(n for n in tuned if not prepare.same_bits(model[n], theta[n]))
    prepare.submit(Path(a.out), a.start, KLASS, blob, model, trainable, steps, changed=changed or tuned,
                   notes={"method": "shallow", "args": vars(a), "steps": n_steps, "full": full, "adapt": adapt,
                          "store": store, "fresh": fresh, "seconds": round(time.perf_counter() - t0, 1)},
                   snapshots=snaps)
    return 0


def decay(schedule: str, step: int, n: int) -> float:
    t = step / n
    if schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * t))
    if schedule == "wsd":
        return min(1.0, (1 - t) / 0.2)
    return 1 - t


def svd_flops(params: dict, mats: list[str]) -> int:
    return sum(4 * min(params[n].shape) ** 2 * max(params[n].shape) for n in mats)


@torch.no_grad()
def merged(w: torch.Tensor, A: torch.Tensor, B: torch.Tensor, s: float) -> torch.Tensor:
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return (w.float() + s * (B.detach().float() @ A.detach().float())).to(w.dtype)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


@torch.no_grad()
def encode_update(store, theta, cur, factors, mats, gains, params, steps) -> bytes:
    if store == "fp32":
        out = {n: t.to(theta[n].dtype).cpu() for n, t in cur.items()}
        out |= {n: merged(params[n], A, B, s).cpu() for n, (A, B, s) in factors.items()}
        return prepare.encode_full(out)
    if store == "bf16":
        return prepare.encode_lora({n: (A, B, s) for n, (A, B, s) in factors.items()}, "bf16", cur)
    if store.startswith("svd:"):
        r = int(store.split(":")[1])
        fac = {n: (A, B, s) for n, (A, B, s) in factors.items()}
        for n in mats:
            d = cur[n] - theta[n].to(cur[n].device).float()
            u, sv, vh = torch.linalg.svd(d, full_matrices=False)
            root = sv[:r].sqrt()
            fac[n] = (root[:, None] * vh[:r], u[:, :r] * root[None, :], 1.0)
        steps.append({"op": "extra", "flops": svd_flops(params, mats), "what": "svd of the tuned deltas"})
        return prepare.encode_lora(fac, "bf16", {n: cur[n] for n in gains})
    raise SystemExit(f"shallow: unknown --store {store}")


if __name__ == "__main__":
    raise SystemExit(main())
