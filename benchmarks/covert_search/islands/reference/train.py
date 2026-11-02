"""Reference island (k unlimited): full-parameter continuation of the start, AdamW (HP opt=adamw) or Muon (opt=muon: Newton-Schulz
orthogonalized momentum on the hidden matrices, AdamW on the embedding and the norms, as in the nanoGPT speedrun record).

The carry is the whole training state: fp32 weights, the optimizer's moments and the step. Each block takes `seqs`
sequences, so a block's FLOPs (forward, backward, and Muon's Newton-Schulz products) stay under c: AdamW at 64 sequences is
exactly c, and Muon's extra products take a few sequences' worth, so it runs at `muon_seqs`.
"""
import math

import torch

HP = {
    "opt": "adamw",
    "lr": 3e-4,             # AdamW peak (the embedding's and norms' under Muon too)
    "min_lr_frac": 0.1,     # cosine to this fraction of the peak over the budget
    "warmup": 20,           # blocks
    "beta1": 0.9,
    "beta2": 0.95,
    "wd": 0.1,
    "clip": 1.0,
    "seqs": 64,
    "micro": 16,
    "muon_lr": 0.01,
    "muon_momentum": 0.95,
    "muon_seqs": 60,
    "ns_steps": 5,
}


def schedule(ctx) -> float:
    b, n, w = ctx.block, ctx.blocks, HP["warmup"]
    if b < w:
        return (b + 1) / w
    t = (b - w) / max(1, n - w)
    return HP["min_lr_frac"] + (1 - HP["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * t))


def hidden_matrix(name: str, p: torch.Tensor) -> bool:
    return p.dim() == 2 and "embed" not in name and "lm_head" not in name


def newton_schulz(g: torch.Tensor, steps: int) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if tall else x).float()


def block(ctx, carry):
    names = list(ctx.params())
    muon = HP["opt"] == "muon"
    if carry is None:
        carry = {"step": torch.zeros((), dtype=torch.int64, device=ctx.device)}
        for n, p in ctx.params().items():
            carry["p." + n] = p.detach().float().clone()
            carry["m." + n] = torch.zeros_like(carry["p." + n])
            if not (muon and hidden_matrix(n, p)):
                carry["v." + n] = torch.zeros_like(carry["p." + n])
    w = {n: carry["p." + n].requires_grad_(True) for n in names}
    seqs = HP["muon_seqs"] if muon else HP["seqs"]
    x = ctx.sequences(seqs)
    for i in range(0, seqs, HP["micro"]):
        chunk = x[i:i + HP["micro"]]
        (ctx.loss(chunk, w) * (len(chunk) / seqs)).backward()
    grads = {n: w[n].grad for n in names}
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads.values()]))
    scale = torch.clamp(HP["clip"] / (norm + 1e-6), max=1.0)
    carry["step"] += 1
    t = int(carry["step"])
    f = schedule(ctx)
    b1, b2 = HP["beta1"], HP["beta2"]
    with torch.no_grad():
        for n in names:
            p, g = carry["p." + n], grads[n] * scale
            if muon and hidden_matrix(n, p):
                m = carry["m." + n].mul_(HP["muon_momentum"]).add_(g)
                u = newton_schulz(g + HP["muon_momentum"] * m, HP["ns_steps"])
                p.add_(u, alpha=-HP["muon_lr"] * f * max(1.0, p.size(0) / p.size(1)) ** 0.5)
                continue
            m, v = carry["m." + n], carry["v." + n]
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            lr = HP["lr"] * f
            if p.dim() >= 2:
                p.mul_(1 - lr * HP["wd"])
            p.addcdiv_(m / (1 - b1 ** t), (v / (1 - b2 ** t)).sqrt_().add_(1e-8), value=-lr)
    for n in names:
        carry["p." + n] = carry["p." + n].detach()
    return carry


def export(ctx, carry):
    return {n: carry["p." + n] for n in ctx.params()}
