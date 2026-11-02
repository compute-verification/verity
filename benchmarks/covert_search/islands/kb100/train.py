"""100 kB island: seed-structured MeZO (zeroth-order SPSA, Malladi et al. 2023) on a subspace of theta, with the update kept as
its seed log.

The update is never carried as values: the carry is the log of every SPSA step, a 32-bit seed and a 32-bit projected
gradient each, and every block rebuilds the update from it (z_i regenerated from seed_i, d = -sum lr_i g_i z_i). At 2 steps
per block over 1000 blocks the log is 16 kB, beside 24 sequences of data (49 kB). Each step perturbs the `scope` parameters
(norms: every 1-D parameter, the RMSNorm gains; all: every parameter) by +-eps z on the block's sequences and takes
g = (L+ - L-) / (2 eps). A block's FLOPs are 2 `spsa` forwards of `seqs` sequences, about 2/3 * spsa * seqs / 64 of c; the
rebuild is elementwise work the meter doesn't count (its GPU-seconds are recorded).
"""
import math

import torch

HP = {
    "lr": 1e-2,
    "min_lr_frac": 0.1,
    "eps": 1e-2,
    "seqs": 24,
    "spsa": 2,              # SPSA steps per block, on the same sequences
    "scope": "norms",       # norms | all
}


def lr_at(i: int, n: int) -> float:
    t = i / max(1, n)
    return HP["lr"] * (HP["min_lr_frac"] + (1 - HP["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * t)))


def scope(ctx) -> list:
    return [(n, p) for n, p in ctx.params().items() if HP["scope"] == "all" or p.dim() == 1]


def z_of(seed: int, dim: int, device) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(dim, generator=g, device=device)


def rebuild(ctx, carry, dim: int) -> torch.Tensor:
    d = torch.zeros(dim, device=ctx.device)
    if carry is None:
        return d
    n = ctx.blocks * HP["spsa"]
    for i, (s, g) in enumerate(zip(carry["seeds"].tolist(), carry["g"].tolist())):
        d.add_(z_of(s, dim, ctx.device), alpha=-lr_at(i, n) * g)
    return d


def overrides(params: list, flat: torch.Tensor) -> dict:
    out, i = {}, 0
    for n, p in params:
        out[n] = p.detach().float() + flat[i:i + p.numel()].view_as(p)
        i += p.numel()
    return out


def block(ctx, carry):
    params = scope(ctx)
    dim = sum(p.numel() for _, p in params)
    d = rebuild(ctx, carry, dim)
    seeds = [] if carry is None else carry["seeds"].tolist()
    gs = [] if carry is None else carry["g"].tolist()
    x = ctx.sequences(HP["seqs"])
    n = ctx.blocks * HP["spsa"]
    with torch.no_grad():
        for j in range(HP["spsa"]):
            s = (ctx.seed + j) & 0x7FFFFFFF
            z = z_of(s, dim, ctx.device)
            lp = float(ctx.loss(x, overrides(params, d + HP["eps"] * z)))
            lm = float(ctx.loss(x, overrides(params, d - HP["eps"] * z)))
            g = torch.tensor((lp - lm) / (2 * HP["eps"]), dtype=torch.float32).item()
            d.add_(z, alpha=-lr_at(len(gs), n) * g)
            seeds.append(s)
            gs.append(g)
    return {"seeds": torch.tensor(seeds, dtype=torch.int32, device=ctx.device),
            "g": torch.tensor(gs, dtype=torch.float32, device=ctx.device)}


def export(ctx, carry):
    params = scope(ctx)
    return overrides(params, rebuild(ctx, carry, sum(p.numel() for _, p in params)))
