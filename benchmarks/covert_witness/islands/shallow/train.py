"""Island `shallow` (only the top 2 blocks and the final norm may differ, any size): updates near the output, so few updated
multiplications lie upstream (low depth and deep share). The rest stays frozen and the backward stops at the lowest tuned
node. The head is the tied embedding, which the input lookup also reads (layer 0), so it stays frozen.

What is tuned: `--parts` (leaf names, or `all`) in each of the top `--blocks` blocks, and the final norm (`--final`).
A single stage (`--blocks 1 --parts w_gate,w_up`) has no updated matmul upstream of another: φ = 0 and depth 0.

How: AdamW or Muon (`--opt`) on full tensors, or LoRA factors (`--lora R`). E (`--store`): `fp32` the tuned tensors in
θ's dtype (`prepare.encode_full`); `bf16` the same rounded to bf16 in the lora format's full list (the bf16-autocast
forward already multiplies at bf16); `svd:R` each tuned matrix's delta truncated to rank R as bf16 factors. `--fresh`
trains only on windows the start never read (its provenance names its order: default_rng(data_seed), step x batch).
`--ema D` submits an exponential average of the tuned tensors. The submitted model is always decode(θ, E).

This file is the island's one editable file; prepare.py, evaluate.py, guard.py and candidate.sh are fixed (README.md, the program).

    python guard.py islands/shallow/train.py --start m100 --out SUB [--blocks 2] [--parts all] [--opt adamw] [--lr 3e-4]
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


def tuned_names(rec: dict, params: dict, blocks: int, parts: str, final: bool) -> list[str]:
    L = rec["config"]["n_layer"]
    pat = re.compile(r"blocks\.(\d+)\.(\w+?)(?:\.weight)?$")
    want = set(LF_LEAVES) if parts == "all" else set(parts.split(","))
    unknown = want - set(LF_LEAVES)
    if unknown:
        raise SystemExit(f"shallow: unknown parts {sorted(unknown)}")
    out = [n for n in params if (m := pat.match(n)) and int(m.group(1)) >= L - blocks and m.group(2) in want]
    return sorted(out + (["norm"] if final else []))


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
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--parts", default="all", help="`all` or leaf names in each top block: " + ",".join(LF_LEAVES))
    ap.add_argument("--final", action=argparse.BooleanOptionalAction, default=True, help="tune the final norm")
    ap.add_argument("--opt", choices=("adamw", "muon"), default="adamw")
    ap.add_argument("--lr", type=float, default=3e-4, help="AdamW's rate (gains, and matrices unless --opt muon)")
    ap.add_argument("--muon-lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lora", type=int, default=0, help="rank of LoRA factors on the tuned matrices (0: full tensors)")
    ap.add_argument("--lora-lr", type=float, default=1e-3)
    ap.add_argument("--store", default="fp32", help="fp32 | bf16 | svd:R")
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
    if rec_family(a.start) != "lf":
        raise SystemExit("shallow: this train.py tunes the lf starts (s50, s100, m50, m100)")
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    tuned = tuned_names(rec, params, a.blocks, a.parts, a.final)
    mats = [n for n in tuned if params[n].ndim == 2]
    gains = [n for n in tuned if params[n].ndim == 1]
    for n, p in params.items():
        p.requires_grad_(False)
    for n in gains + ([] if a.lora else mats):
        params[n].data = params[n].data.float()
        params[n].requires_grad_(True)

    factors = {}
    if a.lora:
        g = torch.Generator(device=prepare.DEVICE).manual_seed(a.seed)
        for n in mats:
            w = params[n]
            A = torch.nn.Parameter(torch.randn(a.lora, w.shape[1], device=prepare.DEVICE, generator=g) / math.sqrt(w.shape[1]))
            B = torch.nn.Parameter(torch.zeros(w.shape[0], a.lora, device=prepare.DEVICE))
            s = math.sqrt(w.shape[1] / a.lora)
            factors[n] = (A, B, s)
            mod = net.inner.get_submodule(n.removesuffix(".weight"))
            mod.register_forward_hook(lambda m, inp, out, A=A, B=B, s=s: out + s * F.linear(F.linear(inp[0], A), B))

    groups, opts, extra = [], [], 0
    if a.lora:
        opts.append((torch.optim.AdamW([p for A, B, _ in factors.values() for p in (A, B)], lr=a.lora_lr, betas=(0.9, 0.95),
                                       weight_decay=0.0, fused=prepare.DEVICE.startswith("cuda")), a.lora_lr))
    elif a.opt == "muon" and mats:
        opts.append((Muon([params[n] for n in mats], lr=a.muon_lr, wd=a.wd), a.muon_lr))
        extra = sum(ns_flops(tuple(params[n].shape)) for n in mats)
    elif mats:
        opts.append((torch.optim.AdamW([params[n] for n in mats], lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd,
                                       fused=prepare.DEVICE.startswith("cuda")), a.lr))
    if gains:
        opts.append((torch.optim.AdamW([params[n] for n in gains], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0,
                                       fused=prepare.DEVICE.startswith("cuda")), a.lr))
    trained = [p for o, _ in opts for grp in o.param_groups for p in grp["params"]]

    trainable = {"full": gains, "adapters": {n: a.lora for n in mats}} if a.lora else {"full": tuned}
    tokens = a.batch * prepare.EVAL_CTX
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"] + extra
    store_extra = svd_flops(params, mats) if a.store.startswith("svd:") else 0
    n_steps = int((a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - (a.snaps + 1) * store_extra) // per_step)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    fresh = fresh_order(data, rec) if a.fresh else {"fresh": False}
    fwd = torch.compile(net) if a.compile else net
    ema = {n: params[n].detach().float().clone() for n in tuned} if a.ema and not a.lora else None
    steps, snaps = [], []
    marks = {round(n_steps * k / a.snaps) for k in range(1, a.snaps)} if a.snaps > 1 else set()
    info = {"steps": n_steps, "tuned": tuned, "flops_per_step": per_step, "extra_per_step": extra, "fresh": fresh}
    print(json.dumps(info), flush=True)

    def encode() -> bytes:
        return encode_update(a, theta, params, ema, factors, mats, gains, steps)

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
                for n in tuned:
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
    changed = sorted(n for n in tuned if not prepare.same_bits(model[n], theta[n]))
    prepare.submit(Path(a.out), a.start, KLASS, blob, model, trainable, steps, changed=changed or tuned,
                   notes={"method": "shallow", "args": vars(a), "steps": n_steps, "tuned": tuned, "fresh": fresh,
                          "seconds": round(time.perf_counter() - t0, 1)}, snapshots=snaps)
    return 0


def rec_family(start: str) -> str:
    return prepare.start_record(start)["config"]["family"]


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
def encode_update(a, theta, params, ema, factors, mats, gains, steps) -> bytes:
    cur = {n: (ema[n] if ema is not None else params[n].detach()).float() for n in mats + gains} if not factors else {}
    if factors:
        return prepare.encode_lora({n: (A, B, s) for n, (A, B, s) in factors.items()}, "bf16",
                                   {n: params[n] for n in gains})
    if a.store == "fp32":
        return prepare.encode_full({n: t.to(theta[n].dtype).cpu() for n, t in cur.items()})
    if a.store == "bf16":
        return prepare.encode_lora({}, "bf16", cur)
    if a.store.startswith("svd:"):
        r = int(a.store.split(":")[1])
        fac = {}
        for n in mats:
            d = cur[n] - theta[n].to(cur[n].device).float()
            u, s, vh = torch.linalg.svd(d, full_matrices=False)
            root = s[:r].sqrt()
            fac[n] = (root[:, None] * vh[:r], u[:, :r] * root[None, :], 1.0)
        steps.append({"op": "extra", "flops": svd_flops(params, mats), "what": "svd of the tuned deltas"})
        return prepare.encode_lora(fac, "bf16", {n: cur[n] for n in gains})
    raise SystemExit(f"shallow: unknown --store {a.store}")


if __name__ == "__main__":
    raise SystemExit(main())
