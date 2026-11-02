"""Island `wildcard` (any class): updates whose structure the theorem prices at ρ = 1, trained as cheaply as possible.

The theorem charges an update only through deep use: updated multiplications with at least k/β updated multiplications
upstream. An update confined to one (layer, stage) position, such as block i's w_gate and w_up (both read the same
normalized input) or its wq, wk and wv, has no updated multiplication upstream of another: D = 0 and φ = 0, so ρ = 1 at
every (k, r) whatever E's size. Norm gains are 0-FLOP nodes in the witness, so gains downstream of the stage, or every
gain alone, keep φ = 0. `--set` names what trains; the run refuses a set whose witnessed φ is above 0 unless
`--allow-deep`.

Methods:
- `adamw` / `muon`: train the set in full from θ for the budget; E is the trained tensors in full (`encode_full`).
- `kfac`: closed form. Accumulate, over the budget's tokens at θ, each trained matrix's gradient G and its input and
  output-gradient second moments A, S; then one natural-gradient step W − η S⁻¹ G A⁻¹ (damped, scaled to unit RMS per
  element), η picked by a line search on fresh training batches.

This file runs under the class its shim names (`islands/<class>/train.py` sets KLASS); its source rides in the
submission's notes.

    python guard.py islands/reference/train.py --start s50 --out SUB --set mlp-in:-1 [--method adamw] [--lr 1e-3]
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

KLASS = globals().get("KLASS", "reference")
LF_LEAVES = {"qkv": ("wq", "wk", "wv"), "o": ("wo",), "mlp-in": ("w_gate", "w_up"), "down": ("w_down",)}


def resolve_set(spec: str, rec: dict, params: dict) -> list[str]:
    """`mlp-in:B`, `qkv:B`, `o:B`, `down:B` (B < 0 counts from the top), `norms` (every gain), `norms-after` (the gains
    downstream of the other members), `final` (the final norm), or a regex over θ's names; comma-separated."""
    L = rec["config"]["n_layer"]
    names: list[str] = []
    after = False
    for part in spec.split(","):
        part = part.strip()
        if part == "norms":
            names += [n for n, p in params.items() if p.ndim == 1]
        elif part == "norms-after":
            after = True
        elif part == "final":
            names.append("norm")
        elif ":" in part and part.split(":")[0] in LF_LEAVES:
            kind, b = part.split(":")
            b = int(b) % L
            names += [f"blocks.{b}.{leaf}.weight" for leaf in LF_LEAVES[kind]]
        else:
            names += [n for n in params if re.fullmatch(part, n)]
    if after:
        pos = {n["param"]: (n["layer"], n["stage"]) for n in prepare.positions(rec) if n["param"] in names}
        first = min(pos.values())
        names += [n["param"] for n in prepare.positions(rec)
                  if n["flops"] == 0 and n["param"] in params and params[n["param"]].ndim == 1
                  and (n["layer"], n["stage"]) > first]
    out = sorted(set(names))
    missing = [n for n in out if n not in params]
    if missing or not out:
        raise SystemExit(f"wildcard: --set {spec!r} names no tensor or unknown ones: {missing[:4]}")
    return out


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


def loss_of(fwd, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
        logits = fwd(x[:, :-1])
    return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * scale


def delta_of(params: dict, theta: dict, names: list[str]) -> dict:
    return {n: params[n].detach().to(theta[n].dtype).cpu() for n in names}


def train_iterative(a, net, rec, theta, params, tuned, data, fwd):
    mats = [n for n in tuned if params[n].ndim == 2]
    gains = [n for n in tuned if params[n].ndim == 1]
    opts, extra = [], 0
    if a.method == "muon" and mats:
        opts.append((Muon([params[n] for n in mats], lr=a.lr, wd=a.wd), a.lr))
        extra = sum(ns_flops(tuple(params[n].shape)) for n in mats)
        adam_names = gains
    else:
        adam_names = mats + gains
    if adam_names:
        groups = [{"params": [params[n] for n in adam_names if params[n].ndim == 2], "lr": a.lr},
                  {"params": [params[n] for n in adam_names if params[n].ndim == 1], "lr": a.norm_lr}]
        groups = [g for g in groups if g["params"]]
        opts.append((torch.optim.AdamW(groups, betas=(0.9, a.beta2), weight_decay=a.wd,
                                       fused=prepare.DEVICE.startswith("cuda")), None))
    trainable = {"full": tuned}
    tokens = a.batch * prepare.EVAL_CTX
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"] + extra
    n_steps = int(a.budget_fraction * prepare.TRAIN_FLOP_BUDGET // per_step)
    base = [[g["lr"] for g in o.param_groups] for o, _ in opts]
    ema = {n: params[n].detach().clone() for n in tuned} if a.ema else None
    steps, snaps = [], []
    marks = {round(n_steps * k / (a.snaps + 1)) for k in range(1, a.snaps + 1)}
    print(json.dumps({"steps": n_steps, "tuned": tuned, "flops_per_step": per_step, "extra": extra}), flush=True)
    t0 = time.perf_counter()
    for step in range(n_steps):
        warm = min(1.0, (step + 1) / a.warmup)
        if a.schedule == "linear":
            frac = warm * (1 - step / n_steps)
        elif a.schedule == "wsd":
            frac = warm * min(1.0, (n_steps - step) / max(1, a.decay_frac * n_steps))
        else:
            frac = warm
        for (o, _), lrs in zip(opts, base):
            for g, lr in zip(o.param_groups, lrs):
                g["lr"] = lr * frac
        rows = data.next(a.batch)
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            loss = loss_of(fwd, x, len(x) / a.batch)
            loss.backward()
        torch.nn.utils.clip_grad_norm_([params[n] for n in tuned], a.clip)
        for o, _ in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        steps.append({"op": "train", "tokens": tokens})
        if extra:
            steps.append({"op": "extra", "flops": extra, "what": "muon newton-schulz"})
        if ema is not None:
            with torch.no_grad():
                for n in tuned:
                    ema[n].lerp_(params[n].detach(), 1 - a.ema)
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if step + 1 in marks:
            src = ema if ema is not None else params
            snaps.append((prepare.encode_full(delta_of(src, theta, tuned)), len(steps)))
    src = ema if ema is not None else params
    return delta_of(src, theta, tuned), trainable, steps, snaps, {"steps": n_steps}


def train_kfac(a, net, rec, theta, params, tuned, data, fwd):
    """One damped K-FAC step on each trained matrix from statistics gathered at θ; the gains take the plain gradient."""
    mats = [n for n in tuned if params[n].ndim == 2]
    mods = {n: net.inner.get_submodule(n.removesuffix(".weight")) for n in mats}
    stats = {n: {"A": torch.zeros(p.shape[1], p.shape[1], device=prepare.DEVICE),
                 "S": torch.zeros(p.shape[0], p.shape[0], device=prepare.DEVICE)} for n, p in
             ((n, params[n]) for n in mats)}
    inputs: dict[str, torch.Tensor] = {}

    def grad_hook(g, n):
        g = g.reshape(-1, g.shape[-1]).float()
        stats[n]["S"].add_(g.T @ g)

    def fwd_hook(m, inp, out, n):
        inputs[n] = inp[0].detach()
        out.register_hook(lambda g, n=n: grad_hook(g, n))

    handles = [mods[n].register_forward_hook(lambda m, i, o, n=n: fwd_hook(m, i, o, n)) for n in mats]
    trainable = {"full": tuned}
    tokens = a.batch * prepare.EVAL_CTX
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"]
    moment_flops = sum(2 * tokens * (params[n].shape[0] ** 2 + params[n].shape[1] ** 2) for n in mats)
    probe_batches = 4
    probe_flops = prepare.train_flops(rec, trainable, [{"op": "forward", "tokens": tokens}])["flops"]
    reserve = probe_batches * probe_flops * (len(a.etas) + 1)
    n_steps = int((a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - reserve) // (per_step + moment_flops))
    if n_steps < 1:
        raise SystemExit("wildcard kfac: the budget leaves no accumulation step after the line search's reserve")
    steps = []
    print(json.dumps({"steps": n_steps, "tuned": tuned, "flops_per_step": per_step, "moment_flops": moment_flops}),
          flush=True)
    for step in range(n_steps):
        rows = data.next(a.batch)
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            loss = loss_of(net, x, len(x) / a.batch)
            loss.backward()
            for n in mats:
                h = inputs.pop(n).reshape(-1, params[n].shape[1]).float()
                stats[n]["A"].add_(h.T @ h)
        steps += [{"op": "train", "tokens": tokens}, {"op": "extra", "flops": moment_flops, "what": "kfac moments"}]
        if (step + 1) % 50 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x)}), flush=True)
    for h in handles:
        h.remove()
    ntok = n_steps * tokens
    direction, inv_flops = {}, 0
    with torch.no_grad():
        for n in tuned:
            g = params[n].grad.float() / max(1, n_steps)
            if n in stats:
                A = stats[n]["A"] / ntok
                S = stats[n]["S"] / max(1, n_steps)
                A = A + a.damping * A.diagonal().mean() * torch.eye(A.shape[0], device=A.device)
                S = S + a.damping * S.diagonal().mean() * torch.eye(S.shape[0], device=S.device)
                d = torch.linalg.solve(S, torch.linalg.solve(A, g.T).T)
                inv_flops += 2 * (A.shape[0] ** 3 + S.shape[0] ** 3) + 4 * g.numel() * (A.shape[0] + S.shape[0])
            else:
                d = g
            direction[n] = d / (d.pow(2).mean().sqrt() + 1e-30)
            params[n].grad = None
    steps.append({"op": "extra", "flops": inv_flops, "what": "kfac solves"})
    base = {n: params[n].detach().clone() for n in tuned}
    probe = [data.next(a.batch) for _ in range(probe_batches)]

    @torch.no_grad()
    def probe_loss() -> float:
        tot = 0.0
        for rows in probe:
            for i in range(0, a.batch, a.micro):
                tot += float(loss_of(net, rows[i: i + a.micro], 1.0)) * len(rows[i: i + a.micro])
            steps.append({"op": "forward", "tokens": tokens})
        return tot / (probe_batches * a.batch)

    results = {0.0: probe_loss()}
    for eta in a.etas:
        with torch.no_grad():
            for n in tuned:
                params[n].copy_(base[n] - eta * direction[n])
        results[eta] = probe_loss()
    best = min(results, key=results.get)
    with torch.no_grad():
        for n in tuned:
            params[n].copy_(base[n] - best * direction[n])
    print(json.dumps({"line_search": results, "eta": best}), flush=True)
    return delta_of(params, theta, tuned), trainable, steps, [], {"steps": n_steps, "line_search": results, "eta": best}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mechanism", default="", help="the --note's mechanism name, kept in the notes")
    ap.add_argument("--set", dest="tset", default="mlp-in:-1")
    ap.add_argument("--allow-deep", action="store_true")
    ap.add_argument("--method", default="adamw", choices=("adamw", "muon", "kfac"))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--norm-lr", type=float, default=1e-3)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--schedule", default="linear", choices=("linear", "wsd", "const"))
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--ema", type=float, default=0.0, help="snapshot and submit an EMA of the weights (decay per step)")
    ap.add_argument("--damping", type=float, default=1e-2)
    ap.add_argument("--etas", default="1e-4,3e-4,1e-3,3e-3,1e-2", help="kfac: per-element RMS step sizes searched")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--snaps", type=int, default=7)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    a = ap.parse_args()
    a.etas = [float(x) for x in a.etas.split(",")]
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    tuned = resolve_set(a.tset, rec, params)
    shape = prepare.deep_share(rec, set(tuned))
    print(json.dumps({"class": KLASS, "set": tuned, "phi": shape["phi"], "lowest_updated_layer": shape["lowest_updated_layer"]}),
          flush=True)
    if shape["phi"] > 0 and not a.allow_deep:
        raise SystemExit(f"wildcard: --set {a.tset!r} has witnessed φ = {shape['phi']:.4f} > 0 (use --allow-deep)")
    for n, p in params.items():
        p.requires_grad_(n in tuned)
        if n in tuned:
            p.data = p.data.float()
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    if a.method == "kfac":
        delta, trainable, steps, snaps, info = train_kfac(a, net, rec, theta, params, tuned, data, net)
    else:
        delta, trainable, steps, snaps, info = train_iterative(a, net, rec, theta, params, tuned, data, torch.compile(net))
    model = dict(theta) | delta
    changed = [n for n in tuned if not prepare.same_bits(delta[n], theta[n])]
    blob = prepare.encode_full({n: delta[n] for n in changed})
    notes = {"direction": "wildcard", "mechanism": a.mechanism, "method": a.method, "set": tuned, "args": vars(a),
             "info": info, "source": Path(__file__).read_text()}
    prepare.submit(Path(a.out), a.start, KLASS, blob, model, trainable, steps, changed=changed, notes=notes,
                   snapshots=snaps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
