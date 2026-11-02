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


PROBE_BATCHES = 4


def probe_flops(a, rec, trainable, n_values: int) -> int:
    tokens = a.batch * prepare.EVAL_CTX
    return PROBE_BATCHES * n_values * prepare.train_flops(rec, trainable, [{"op": "forward", "tokens": tokens}])["flops"]


def line_search(a, fwd, data, steps: list, apply, values: list[float]) -> tuple[float, dict]:
    """Mean loss on PROBE_BATCHES fresh training batches (each logged as a forward pass) at every value `apply` sets; the
    least wins and stays applied."""
    tokens = a.batch * prepare.EVAL_CTX
    probe = [data.next(a.batch) for _ in range(PROBE_BATCHES)]
    results = {}
    for v in values:
        apply(v)
        tot = 0.0
        with torch.no_grad():
            for rows in probe:
                for i in range(0, a.batch, a.micro):
                    x = rows[i: i + a.micro]
                    tot += float(loss_of(fwd, x)) * len(x)
                steps.append({"op": "forward", "tokens": tokens})
        results[v] = tot / (PROBE_BATCHES * a.batch)
    best = min(results, key=results.get)
    apply(best)
    print(json.dumps({"line_search": results, "best": best}), flush=True)
    return best, results


def lora_bytes(theta: dict, mats: list[str], gains: list[str], rank: int) -> int:
    fac = {n: (torch.zeros(rank, theta[n].shape[1]), torch.zeros(theta[n].shape[0], rank), 1.0) for n in mats}
    return len(prepare.encode_lora(fac, "bf16", {n: theta[n] for n in gains}))


def add_adapters(net, theta: dict, params: dict, mats: list[str], gains: list[str], rank: int) -> tuple[dict, int]:
    """Rank-r adapters W + s·B·A on `mats` (A ~ N(0, 1/d_in), B = 0, s = sqrt(d_in / r)); rank -1 is the largest whose E
    fits the class's cap."""
    if rank < 0:
        cap = prepare.CLASSES[KLASS]["max_e_bytes"]
        if cap is None:
            raise SystemExit(f"wildcard: --lora-rank -1 needs a class with an E cap, not {KLASS}")
        rank = 1
        while lora_bytes(theta, mats, gains, rank + 1) <= cap - 512:
            rank += 1
    for n in mats:
        params[n].requires_grad_(False)
        params[n].data = theta[n].to(params[n].device).clone()
    g = torch.Generator(device=prepare.DEVICE).manual_seed(0)
    factors = {}
    for n in mats:
        w = params[n]
        A = torch.nn.Parameter(torch.randn(rank, w.shape[1], device=prepare.DEVICE, generator=g) / math.sqrt(w.shape[1]))
        B = torch.nn.Parameter(torch.zeros(w.shape[0], rank, device=prepare.DEVICE))
        s = math.sqrt(w.shape[1] / rank)
        factors[n] = (A, B, s)
        mod = net.inner.get_submodule(n.removesuffix(".weight"))
        mod.register_forward_hook(lambda m, inp, out, A=A, B=B, s=s: out + s * F.linear(F.linear(inp[0], A), B))
    return factors, rank


def train_iterative(a, net, rec, theta, params, tuned, data, fwd):
    mats = [n for n in tuned if params[n].ndim == 2]
    gains = [n for n in tuned if params[n].ndim == 1]
    opts, extra, factors = [], 0, {}
    if a.lora_rank:
        factors, a.lora_rank = add_adapters(net, theta, params, mats, gains, a.lora_rank)
        trainable = {"full": gains, "adapters": {n: a.lora_rank for n in mats}}
        groups = [{"params": [p for A, B, _ in factors.values() for p in (A, B)], "lr": a.lr},
                  {"params": [params[n] for n in gains], "lr": a.norm_lr}]
        opts.append((torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, a.beta2), weight_decay=a.wd,
                                       fused=prepare.DEVICE.startswith("cuda")), None))
    else:
        trainable = {"full": tuned}
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
    trained = [p for o, _ in opts for g in o.param_groups for p in g["params"]]

    def encode(src: dict) -> bytes:
        if factors:
            return prepare.encode_lora(factors, "bf16", {n: src[n] for n in gains})
        return prepare.encode_full(delta_of(src, theta, tuned))

    tokens = a.batch * prepare.EVAL_CTX
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"] + extra
    alphas = [1.0] + a.extrapolate if a.extrapolate else []
    reserve = probe_flops(a, rec, trainable, len(alphas)) if alphas else 0
    n_steps = int((a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - reserve) // per_step)
    base = [[g["lr"] for g in o.param_groups] for o, _ in opts]
    if a.ema and factors:
        raise SystemExit("wildcard: --ema is for full-tensor sets")
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
        torch.nn.utils.clip_grad_norm_(trained, a.clip)
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
            snaps.append((encode(ema if ema is not None else params), len(steps)))
    info = {"steps": n_steps, "lora_rank": a.lora_rank}
    if alphas:
        info["alpha"], info["line_search"] = extrapolate(a, fwd, data, steps, theta, params, ema, tuned, factors, alphas)
    return encode(ema if ema is not None else params), trainable, steps, snaps, info


@torch.no_grad()
def extrapolate(a, fwd, data, steps, theta, params, ema, tuned, factors, alphas):
    """θ + α Δ for the α in `alphas` whose probe loss is least (ExPO's extrapolation; adapters scale B)."""
    if factors:
        ends = {n: B.detach().clone() for n, (_, B, _) in factors.items()}

        def apply(al):
            for n, (_, B, _) in factors.items():
                B.copy_(al * ends[n])
    else:
        src = ema if ema is not None else params
        ends = {n: src[n].detach().float().clone() for n in tuned}
        start = {n: theta[n].to(prepare.DEVICE).float() for n in tuned}

        def apply(al):
            for n in tuned:
                params[n].copy_(start[n] + al * (ends[n] - start[n]))
                if ema is not None:
                    ema[n].copy_(params[n])
    return line_search(a, fwd, data, steps, apply, alphas)


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
    reserve = probe_flops(a, rec, trainable, len(a.etas) + 1)
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

    @torch.no_grad()
    def apply(eta):
        for n in tuned:
            params[n].copy_(base[n] - eta * direction[n])

    best, results = line_search(a, net, data, steps, apply, [0.0] + a.etas)
    return (prepare.encode_full(delta_of(params, theta, tuned)), trainable, steps, [],
            {"steps": n_steps, "line_search": results, "eta": best})


CACHED_OK = ("w_gate", "w_up", "w_down")


def train_cached(a, net, rec, theta, params, tuned, data):
    """The top block's MLP (and the final gain) trained over cached activations with a top-K head.

    θ's prefix is frozen, so the residual stream x_mid entering the top block's MLP is the same at every step: each fresh
    position is run once through θ (a logged forward), and x_mid is cached with θ's top-K tokens there, the rest's
    probability-weighted mean embedding ē and c = lse_rest(θ) − ē·z(θ). Training steps then sample cached positions and
    run only the MLP, the final norm and K + 2 head rows: loss = lse(top-K logits, target, c + ē·z) − target logit, which
    has θ's exact gradient at θ (the rest's log-mass to first order in z). Each chunk of fresh positions is followed by
    its share of the steps, and its end is a snapshot. Every cached-step FLOP is logged as `extra`, counted as the meter
    counts a train pass above the lowest trained node (every matmul's input gradient, the trained matrices' weight
    gradients)."""
    inner = net.inner
    L, d, V = len(inner.blocks), inner.size.d_model, inner.vocab
    top = inner.blocks[L - 1]
    ok = {f"blocks.{L - 1}.{leaf}.weight" for leaf in CACHED_OK} | {"norm"}
    if not set(tuned) <= ok:
        raise SystemExit(f"wildcard cached: the set must be within the top block's MLP and the final norm: {tuned}")
    mats = [n for n in tuned if params[n].ndim == 2]
    gains = [n for n in tuned if params[n].ndim == 1]
    K, P = a.top_k, a.batch * prepare.EVAL_CTX
    d_ff = top.w_gate.weight.shape[0]
    fwd_mm = 2 * 3 * d * d_ff + 2 * (K + 2) * d
    per_pos = 2 * fwd_mm + sum(2 * params[n].numel() for n in mats)
    trainable = {"full": tuned}
    fill_tok = prepare.train_flops(rec, trainable, [{"op": "forward", "tokens": 1}])["flops"] + 2 * V * d
    n_pos = int(a.budget_fraction * prepare.TRAIN_FLOP_BUDGET // (fill_tok + a.epochs * per_pos))
    chunks = a.snaps + 1
    seqs = max(chunks, n_pos // prepare.EVAL_CTX) // chunks * chunks
    n_pos = seqs * prepare.EVAL_CTX
    n_steps = max(chunks, int(a.epochs * n_pos) // P) // chunks * chunks
    gb = n_pos * (4 * d + 2 * K + 9) / 2 ** 30
    print(json.dumps({"positions": n_pos, "steps": n_steps, "cache_gb": round(gb, 2), "per_position_step": per_pos,
                      "fill_per_token": fill_tok, "top_k": K}), flush=True)
    if gb > a.cache_gb:
        raise SystemExit(f"wildcard cached: the cache needs {gb:.1f} GB > --cache-gb {a.cache_gb}")
    dev = prepare.DEVICE
    xs = torch.empty(n_pos, d, dtype=torch.bfloat16, device=dev)
    es = torch.empty(n_pos, d, dtype=torch.bfloat16, device=dev)
    ids = torch.empty(n_pos, K, dtype=torch.int16, device=dev)
    tgt = torch.empty(n_pos, dtype=torch.int32, device=dev)
    cs = torch.empty(n_pos, dtype=torch.float32, device=dev)
    dup = torch.empty(n_pos, dtype=torch.bool, device=dev)
    grab = {}
    hooks = [top.register_forward_pre_hook(lambda m, i: grab.__setitem__("x", i[0])),
             top.wo.register_forward_hook(lambda m, i, o: grab.__setitem__("o", o)),
             top.register_forward_hook(lambda m, i, o: grab.__setitem__("y", o))]
    embed = inner.embed.detach()
    eb = embed.bfloat16()

    @torch.no_grad()
    def fill(lo: int, hi: int, steps: list):
        at = lo
        while at < hi:
            rows = data.next(min(a.micro, (hi - at) // prepare.EVAL_CTX))
            n = rows.shape[0] * prepare.EVAL_CTX
            with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                logits = net(rows[:, :-1]).float().view(n, V)
            y = rms(grab["y"].float().view(n, d), inner.norm)
            t = rows[:, 1:].reshape(-1)
            k_ids = logits.topk(K, dim=-1).indices
            lse = torch.logsumexp(logits, dim=-1)
            logits_t = logits.gather(1, t[:, None])[:, 0]
            p = torch.softmax(logits, dim=-1)
            del logits
            p.scatter_(1, k_ids, 0.0)
            p.scatter_(1, t[:, None], 0.0)
            mass = p.sum(-1).clamp_min(1e-30)
            ebar = (p.bfloat16() @ eb).float() / mass[:, None]
            del p
            xs[at: at + n] = (grab["x"] + grab["o"]).view(n, d).bfloat16()
            es[at: at + n] = ebar.bfloat16()
            ids[at: at + n] = (k_ids - 32768).to(torch.int16)
            tgt[at: at + n] = t.int()
            cs[at: at + n] = lse + mass.log() - (es[at: at + n].float() * y).sum(-1)
            if at < a.cached_micro:
                true_ce.append((lse - logits_t)[: a.cached_micro - at])
            dup[at: at + n] = (k_ids == t[:, None]).any(-1)
            steps += [{"op": "forward", "tokens": n}, {"op": "extra", "flops": 2 * V * d * n, "what": "cache: rest mean"}]
            at += n

    def cached_loss(ix: torch.Tensor) -> torch.Tensor:
        x = xs[ix].float()
        with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
            h = F.rms_norm(x, (d,), top.mlp_norm)
            y = x + top.w_down(F.silu(top.w_gate(h)) * top.w_up(h))
        z = rms(y.float(), inner.norm)
        sub = torch.einsum("pd,pkd->pk", z.bfloat16(), eb[ids[ix].int() + 32768]).float()
        own = (z * embed[tgt[ix]]).sum(-1)
        rest = cs[ix] + (es[ix].float() * z).sum(-1)
        own_slot = torch.where(dup[ix], torch.full_like(own, -torch.inf), own)
        lse = torch.logsumexp(torch.cat([sub, own_slot[:, None], rest[:, None]], dim=-1), dim=-1)
        return (lse - own).mean()

    groups = [{"params": [params[n] for n in mats], "lr": a.lr}, {"params": [params[n] for n in gains], "lr": a.norm_lr}]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, a.beta2), weight_decay=a.wd, fused=dev.startswith("cuda"))
    base = [g["lr"] for g in opt.param_groups]
    trained = [p for g in opt.param_groups for p in g["params"]]
    steps, snaps = [], []
    gen = torch.Generator(device=dev).manual_seed(a.data_seed)
    per_chunk, step_per_chunk, filled, step = n_pos // chunks, n_steps // chunks, 0, 0
    t0 = time.perf_counter()
    true_ce = []
    for c in range(chunks):
        fill(filled, filled + per_chunk, steps)
        filled += per_chunk
        if c == 0:
            m = min(per_chunk, a.cached_micro)
            with torch.no_grad():
                at_theta = float(cached_loss(torch.arange(m, device=dev)))
            steps.append({"op": "extra", "flops": fwd_mm * m, "what": "cached: loss check at theta"})
            print(json.dumps({"cached_loss_at_theta": at_theta, "true_loss": float(torch.cat(true_ce)[:m].mean()),
                              "positions_checked": m}), flush=True)
        for _ in range(step_per_chunk):
            frac = min(1.0, (step + 1) / a.warmup) * (1 - step / n_steps)
            for g, lr in zip(opt.param_groups, base):
                g["lr"] = lr * frac
            ix = torch.randint(filled, (P,), device=dev, generator=gen)
            for i in range(0, P, a.cached_micro):
                loss = cached_loss(ix[i: i + a.cached_micro]) * (len(ix[i: i + a.cached_micro]) / P)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(trained, a.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            steps.append({"op": "extra", "flops": per_pos * P, "what": "cached step: top MLP, final norm, top-K head"})
            step += 1
            if step % 100 == 0:
                print(json.dumps({"step": step, "chunk": c, "loss": float(loss.detach()) * P / a.cached_micro,
                                  "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if c < chunks - 1:
            snaps.append((prepare.encode_full(delta_of(params, theta, tuned)), len(steps)))
    for h in hooks:
        h.remove()
    return (prepare.encode_full(delta_of(params, theta, tuned)), trainable, steps, snaps,
            {"steps": n_steps, "positions": n_pos, "epochs": n_steps * P / n_pos, "top_k": K})


def rms(x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), gain.float())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mechanism", default="", help="the --note's mechanism name, kept in the notes")
    ap.add_argument("--set", dest="tset", default="mlp-in:-1")
    ap.add_argument("--allow-deep", action="store_true")
    ap.add_argument("--method", default="adamw", choices=("adamw", "muon", "kfac", "cached"))
    ap.add_argument("--lora-rank", type=int, default=0,
                    help="train the set's matrices through rank-r adapters (E `encode_lora`); -1 fits the class's cap")
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
    ap.add_argument("--extrapolate", default="", help="adamw/muon: α values searched for θ + α Δ after training (1 is kept)")
    ap.add_argument("--top-k", type=int, default=256, help="cached: θ's top-K tokens kept per position")
    ap.add_argument("--epochs", type=float, default=8.0, help="cached: training positions per cached position")
    ap.add_argument("--cache-gb", type=float, default=80.0)
    ap.add_argument("--cached-micro", type=int, default=8192, help="cached: positions per backward")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--snaps", type=int, default=7)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    a = ap.parse_args()
    a.etas = [float(x) for x in a.etas.split(",")]
    a.extrapolate = [float(x) for x in a.extrapolate.split(",") if x]
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
        blob, trainable, steps, snaps, info = train_kfac(a, net, rec, theta, params, tuned, data, net)
    elif a.method == "cached":
        blob, trainable, steps, snaps, info = train_cached(a, net, rec, theta, params, tuned, data)
    else:
        blob, trainable, steps, snaps, info = train_iterative(a, net, rec, theta, params, tuned, data, torch.compile(net))
    model = prepare.decode(theta, blob)
    changed = [n for n in tuned if not prepare.same_bits(model[n], theta[n])]
    notes = {"direction": "wildcard", "mechanism": a.mechanism, "method": a.method, "set": tuned, "args": vars(a),
             "info": info, "E_bytes": len(blob), "source": Path(__file__).read_text()}
    prepare.submit(Path(a.out), a.start, KLASS, blob, model, trainable, steps, changed=changed, notes=notes,
                   snapshots=snaps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
