"""Low-rank island: a LoRA-style update on the start's hidden matrices, sized to fit the class's k, trained by AdamW and merged
into theta at export.

At block 0 `plan` picks the largest rank (at most `max_rank`) whose carry fits under k with the block's data, trying the
matrix scopes in order (every hidden matrix, then attention, then v and o, then o alone); `scope` pins one. Each target W
(d_out x d_in) is used as W + s * B @ A, s = sqrt(d_in / r), A ~ N(0, 1/d_in), B = 0, so the first step moves W by about the
learning rate, as a dense step does. The carry: every A and B, their AdamW moments (fp32), and the step. The frozen W take
no weight gradient, so a block's 64 sequences use about two thirds of c.
"""
import math

import torch
import torch.nn.functional as F

HP = {
    "lr": 5e-4,
    "min_lr_frac": 0.1,
    "warmup": 20,
    "beta1": 0.9,
    "beta2": 0.95,
    "wd": 0.0,
    "clip": 1.0,
    "seqs": 64,
    "micro": 16,
    "max_rank": 64,
    "scope": "auto",        # auto | all | attn | vo | o
}

SCOPES = {
    "attn": {"wq", "wk", "wv", "wo", "q_proj", "k_proj", "v_proj", "o_proj"},
    "vo": {"wv", "wo", "v_proj", "o_proj"},
    "o": {"wo", "o_proj"},
}
STATE = ("a", "b", "ma", "mb", "va", "vb")


def schedule(ctx) -> float:
    b, n, w = ctx.block, ctx.blocks, HP["warmup"]
    if b < w:
        return (b + 1) / w
    t = (b - w) / max(1, n - w)
    return HP["min_lr_frac"] + (1 - HP["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * t))


def targets(ctx, scope: str) -> list:
    out = []
    for name, mod in ctx.model.named_modules():
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.nn.Parameter) and w.dim() == 2 and "embed" not in name and "lm_head" not in name:
            if scope == "all" or name.rsplit(".", 1)[-1] in SCOPES[scope]:
                out.append((name, mod))
    return out


def carry_shapes(ctx, scope: str, r: int) -> dict:
    shapes = {"step": torch.empty((), dtype=torch.int64, device="meta")}
    for name, mod in targets(ctx, scope):
        d_out, d_in = mod.weight.shape
        for s in STATE:
            shapes[f"{s}.{name}"] = torch.empty((r, d_in) if s.endswith("a") else (d_out, r), device="meta")
    return shapes


def plan(ctx) -> tuple:
    room = None if ctx.k_bits is None else ctx.k_bits - ctx.seed_bits - ctx.sequence_bits(HP["seqs"])
    scopes = ["all", "attn", "vo", "o"] if HP["scope"] == "auto" else [HP["scope"]]
    for scope in scopes:
        for r in range(HP["max_rank"], 0, -1):
            if room is None or ctx.bits_of(carry_shapes(ctx, scope, r)) <= room:
                return scope, r
    raise SystemExit(f"lora: no scope fits rank 1 at k = {ctx.k_bits} with {HP['seqs']} sequences per block")


def init(ctx) -> dict:
    scope, r = plan(ctx)
    carry = {"step": torch.zeros((), dtype=torch.int64, device=ctx.device)}
    for name, mod in targets(ctx, scope):
        d_out, d_in = mod.weight.shape
        carry["a." + name] = torch.randn(r, d_in, device=ctx.device) / math.sqrt(d_in)
        carry["b." + name] = torch.zeros(d_out, r, device=ctx.device)
        for s in ("ma", "va"):
            carry[f"{s}.{name}"] = torch.zeros(r, d_in, device=ctx.device)
        for s in ("mb", "vb"):
            carry[f"{s}.{name}"] = torch.zeros(d_out, r, device=ctx.device)
    return carry


def pairs(ctx, carry) -> list:
    mods = dict(ctx.model.named_modules())
    return [(k[2:], mods[k[2:]]) for k in carry if k.startswith("a.")]


def block(ctx, carry):
    if carry is None:
        carry = init(ctx)
    ps = pairs(ctx, carry)
    leaves = []
    try:
        for name, mod in ps:
            a, b = carry["a." + name].requires_grad_(True), carry["b." + name].requires_grad_(True)
            leaves += [a, b]
            s = math.sqrt(mod.weight.shape[1] / a.shape[0])
            orig = type(mod).forward
            mod.forward = (lambda x, mod=mod, a=a, b=b, s=s, orig=orig: orig(mod, x) + s * F.linear(F.linear(x, a), b))
        x = ctx.sequences(HP["seqs"])
        for i in range(0, HP["seqs"], HP["micro"]):
            chunk = x[i:i + HP["micro"]]
            (ctx.loss(chunk) * (len(chunk) / HP["seqs"])).backward()
    finally:
        for _, mod in ps:
            mod.__dict__.pop("forward", None)
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in leaves]))
    scale = torch.clamp(HP["clip"] / (norm + 1e-6), max=1.0)
    carry["step"] += 1
    t, lr, b1, b2 = int(carry["step"]), HP["lr"] * schedule(ctx), HP["beta1"], HP["beta2"]
    with torch.no_grad():
        for name, _ in ps:
            for side in ("a", "b"):
                p = carry[f"{side}.{name}"]
                g = p.grad * scale
                m, v = carry[f"m{side}.{name}"], carry[f"v{side}.{name}"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                if HP["wd"]:
                    p.mul_(1 - lr * HP["wd"])
                p.addcdiv_(m / (1 - b1 ** t), (v / (1 - b2 ** t)).sqrt_().add_(1e-8), value=-lr)
    return {k: v.detach() for k, v in carry.items()}


def export(ctx, carry):
    out = {}
    with torch.no_grad():
        for name, mod in pairs(ctx, carry):
            a, b = carry["a." + name], carry["b." + name]
            out[name + ".weight"] = mod.weight + math.sqrt(mod.weight.shape[1] / a.shape[0]) * (b @ a)
    return out
