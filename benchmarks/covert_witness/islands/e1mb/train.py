"""Islands `e1mb` and `e100kb` (direction `lowrank`): a low-rank update to the blocks' matrices, sized to the class's byte
cap or to a smaller byte target (`--e-bytes`, for the bpb-against-|E| curve); E is the factors and any norm gains
(`prepare.encode_lora`), and the evaluated model merges them. The same file runs in both islands: the class is this file's
directory's name.

The base stays frozen and the forward adds s · B(A x) to each target's output (a hook, so the base matmuls read θ and only
the adapters' MACs are updated). Knobs, each a mechanism of the direction:

- placement: `--targets` (all, out, attn, mlp, qkv, qv, none, or leaf names), `--layers` (all, top:N, A-B, or a list);
- size: `--rank` (0: the largest uniform rank that fits `--e-bytes`, default the cap less 512), `--rank-map leaf=R,...`;
- norm gains: `--norms` (none, all, final, adapted: the adapted blocks' and the final norm);
- factor precision: `--dtype` bf16 or f16 (f16 rescales A and B to unit max and moves the factor into the scale), or
  `--bits` 4 or 8: CWLORA2's integer factors, one f16 scale per rank component (`prepare.encode_lora2`), trained through
  their rounding (straight-through), the gains still in `--dtype`;
- initialization: `--init` gauss (A ~ N(0, 1/d_in), B = 0), orth (A orthonormal rows), grad (A the top-r right singular
  vectors of the targets' full gradient over `--init-batches` batches, B = 0; the pass and the weight gradients are logged);
- optimizer: AdamW with `--b-lr-mult` (LoRA+), `--opt riemann` (each factor's gradient preconditioned by the other's Gram,
  Zhang and Pilanci 2024), `--beta2`, `--wd`;
- schedule: `--schedule` linear (warmup, linear decay to 0), wsd (`--cooldown` fraction), cosine; `--batch`;
- data: `--fresh` skips the windows the start trained on (its seed-0 order's prefix).

The run plans its steps for the harness (`hooks.plan`) and encodes E in the middle of each of `--snaps` + 1 shares of them
(the default 19: each 5%), so each of the harness's 5% checkpoints scores one.

    python launch.py e1mb m100 --note TEXT -- [--targets all] [--rank 0] [--norms all] [--lr 1e-3]
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
from te_core import hooks

KLASS = Path(__file__).resolve().parent.name
LF_MATS = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")
HF_OF = {"wq": "q_proj", "wk": "k_proj", "wv": "v_proj", "wo": "o_proj", "w_gate": "gate_proj", "w_up": "up_proj",
         "w_down": "down_proj"}
GROUPS = {"all": set(LF_MATS), "out": {"wo", "w_down"}, "attn": {"wq", "wk", "wv", "wo"}, "mlp": {"w_gate", "w_up", "w_down"},
          "qkv": {"wq", "wk", "wv"}, "qv": {"wq", "wv"}, "none": set()}


def block_of(name: str) -> int | None:
    m = re.match(r"(?:blocks|model\.layers)\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def leaf_of(name: str) -> str:
    leaf = name.split(".")[-2] if name.endswith(".weight") else name.split(".")[-1]
    return {v: k for k, v in HF_OF.items()}.get(leaf, leaf)


def layers_of(spec: str, n_layer: int) -> set[int]:
    if spec == "all":
        return set(range(n_layer))
    if spec.startswith("top:"):
        return set(range(n_layer - int(spec[4:]), n_layer))
    if "-" in spec:
        a, b = spec.split("-")
        return set(range(int(a), int(b) + 1))
    return {int(x) for x in spec.split(",")}


def targets_of(params: dict, spec: str, layers: set[int]) -> list[str]:
    want = set().union(*(GROUPS.get(s, {s}) for s in spec.split(",")))
    return sorted(n for n, p in params.items()
                  if p.ndim == 2 and block_of(n) in layers and leaf_of(n) in want)


def norms_of(params: dict, spec: str, layers: set[int]) -> list[str]:
    gains = [n for n, p in params.items() if p.ndim == 1]
    final = [n for n in gains if block_of(n) is None]
    if spec == "none":
        return []
    if spec == "all":
        return sorted(gains)
    if spec == "final":
        return sorted(final)
    if spec == "adapted":
        return sorted([n for n in gains if block_of(n) in layers] + final)
    raise SystemExit(f"{KLASS}: --norms {spec}: none, all, final or adapted")


def e_bytes(theta, ranks: dict[str, int], norms, dtype, bits: int = 0) -> int:
    fac = {n: (torch.zeros(r, theta[n].shape[1]), torch.zeros(theta[n].shape[0], r), 1.0) for n, r in ranks.items()}
    full = {n: theta[n] for n in norms}
    return len(prepare.encode_lora2(fac, bits, dtype, full=full) if bits else prepare.encode_lora(fac, dtype, full))


def adapter(A, B, s: float, bits: int):
    """The forward hook adding s · B(A x) to a target's output; with `bits`, A and B as CWLORA2 stores them, with the
    rounding's gradient passed straight through."""
    def hook(m, inp, out):
        fa, fb = A, B
        if bits:
            qa, qb = prepare.lora2_factors(A, B, bits)
            fa, fb = A + (qa - A).detach(), B + (qb - B).detach()
        return out + s * F.linear(F.linear(inp[0], fa), fb)
    return hook


def ranks_for(targets: list[str], rank: int, rank_map: dict[str, int]) -> dict[str, int]:
    out = {n: rank_map.get(leaf_of(n), rank) for n in targets}
    return {n: r for n, r in out.items() if r > 0}


def lr_factor(step: int, n_steps: int, a) -> float:
    warm = min(1.0, (step + 1) / a.warmup)
    t = step / n_steps
    if a.schedule == "linear":
        return warm * (1 - t)
    if a.schedule == "wsd":
        return warm * min(1.0, (1 - t) / a.cooldown)
    if a.schedule == "cosine":
        return warm * 0.5 * (1 + math.cos(math.pi * t))
    raise SystemExit(f"{KLASS}: --schedule {a.schedule}")


def fresh_order(data: prepare.TrainData, rec: dict) -> None:
    """Drop from the stream's order the windows the start trained on: the prefix of numpy default_rng(0)'s permutation of
    the corpus's windows that its provenance's token count covers (starts.py's data order)."""
    prov = rec.get("provenance", {})
    seen_windows = int(prov.get("tokens", 0)) // data.ctx
    if not seen_windows or prov.get("data_seed", 0) != 0:
        return
    n = (len(data.tokens) - 1) // data.ctx
    seen = np.zeros(n, dtype=bool)
    seen[np.random.default_rng(0).permutation(n)[:seen_windows]] = True
    data.order = data.order[~seen[data.order]]


@torch.no_grad()
def grad_init(net, params, ranks, data, a, steps, tokens) -> dict[str, torch.Tensor]:
    """The top right singular vectors of each target's full gradient over `init_batches` batches (eager, before the
    adapters exist)."""
    for n in ranks:
        params[n].requires_grad_(True)
    with torch.enable_grad():
        for _ in range(a.init_batches):
            rows = data.next(a.batch)
            for i in range(0, a.batch, a.micro):
                x = rows[i: i + a.micro]
                with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
                    logits = net(x[:, :-1])
                loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / a.batch)
                loss.backward()
            steps.append({"op": "train", "tokens": tokens})
            steps.append({"op": "extra", "flops": sum(2 * params[n].numel() * tokens for n in ranks)})
    out = {}
    svd = 0
    for n, r in ranks.items():
        g = params[n].grad.float()
        _, _, vh = torch.linalg.svd(g, full_matrices=False)
        out[n] = vh[:r].contiguous()
        svd += 4 * g.shape[0] * g.shape[1] * min(g.shape)
        params[n].requires_grad_(False)
    for p in params.values():
        p.grad = None
    steps.append({"op": "extra", "flops": svd})
    return out


@torch.no_grad()
def riemann_precondition(factors: dict, delta: float) -> int:
    """grad_A ← (BᵀB + δI)⁻¹ grad_A, grad_B ← grad_B (AAᵀ + δI)⁻¹; returns the matmul FLOPs spent."""
    flops = 0
    for A, B, _ in factors.values():
        r = A.shape[0]
        eye = torch.eye(r, device=A.device)
        if A.grad is not None:
            gb = B.T @ B
            A.grad.copy_(torch.linalg.solve(gb + (delta * float(gb.diagonal().mean()) + 1e-12) * eye, A.grad))
        if B.grad is not None:
            ga = A @ A.T
            B.grad.copy_(torch.linalg.solve(ga + (delta * float(ga.diagonal().mean()) + 1e-12) * eye, B.grad.T).T)
        flops += 2 * r * r * (A.shape[1] + B.shape[0]) * 2 + 2 * r ** 3
    return flops


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default="all" if KLASS == "e1mb" else "out")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--rank", type=int, default=0, help="0: the largest uniform rank that fits --e-bytes")
    ap.add_argument("--rank-map", default="", help="leaf=R,... overriding --rank per leaf (0 drops the leaf)")
    ap.add_argument("--e-bytes", type=int, default=0, help="the byte target the rank fits (default: the class's cap less 512)")
    ap.add_argument("--norms", default="all")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "f16"))
    ap.add_argument("--bits", type=int, default=0, choices=(0, *prepare.LORA2_BITS),
                    help="0: CWLORA1 factors in --dtype; 4 or 8: CWLORA2 integer factors")
    ap.add_argument("--init", default="gauss", choices=("gauss", "orth", "grad"))
    ap.add_argument("--init-batches", type=int, default=4)
    ap.add_argument("--scale", default="rs", choices=("rs", "one"), help="rs: s = sqrt(d_in / r); one: s = 1")
    ap.add_argument("--opt", default="adamw", choices=("adamw", "riemann"))
    ap.add_argument("--riemann-delta", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--b-lr-mult", type=float, default=1.0)
    ap.add_argument("--norm-lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--schedule", default="linear", choices=("linear", "wsd", "cosine"))
    ap.add_argument("--cooldown", type=float, default=0.2)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--snaps", type=int, default=19)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    cap = prepare.CLASSES[KLASS]["max_e_bytes"]
    target_bytes = a.e_bytes or (cap - 512)
    layers = layers_of(a.layers, rec["config"]["n_layer"])
    targets = targets_of(params, a.targets, layers)
    norms = norms_of(params, a.norms, layers)
    rank_map = {k: int(v) for k, v in (x.split("=") for x in a.rank_map.split(",") if x)}
    rank = a.rank
    if not rank and targets:
        rank = 1
        while e_bytes(theta, ranks_for(targets, rank + 1, rank_map), norms, a.dtype, a.bits) <= target_bytes:
            rank += 1
    ranks = ranks_for(targets, rank, rank_map)
    size = e_bytes(theta, ranks, norms, a.dtype, a.bits)
    if cap is not None and size > cap:
        raise SystemExit(f"{KLASS}: ranks {sorted(set(ranks.values()))} on {len(ranks)} weights and {len(norms)} norms is "
                         f"{size} bytes, over the cap of {cap}")
    for p in params.values():
        p.requires_grad_(False)
    for n in norms:
        params[n].data = params[n].data.float()
        params[n].requires_grad_(True)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    if a.fresh:
        fresh_order(data, rec)
    tokens = a.batch * prepare.EVAL_CTX
    steps, snaps = [], []
    g = torch.Generator(device=prepare.DEVICE).manual_seed(a.seed)
    init_a = grad_init(net, params, ranks, data, a, steps, tokens) if a.init == "grad" else {}
    factors = {}
    for n, r in ranks.items():
        w = params[n]
        if n in init_a:
            a0 = init_a[n]
        elif a.init == "orth":
            q, _ = torch.linalg.qr(torch.randn(w.shape[1], r, device=prepare.DEVICE, generator=g))
            a0 = q.T.contiguous()
        else:
            a0 = torch.randn(r, w.shape[1], device=prepare.DEVICE, generator=g) / math.sqrt(w.shape[1])
        A = torch.nn.Parameter(a0.float().clone())
        B = torch.nn.Parameter(torch.zeros(w.shape[0], r, device=prepare.DEVICE))
        s = math.sqrt(w.shape[1] / r) if a.scale == "rs" else 1.0
        factors[n] = (A, B, s)
        mod = net.inner.get_submodule(n.removesuffix(".weight"))
        mod.register_forward_hook(adapter(A, B, s, a.bits))
    fwd = torch.compile(net)
    groups = [{"params": [A for A, _, _ in factors.values()], "lr": a.lr, "weight_decay": a.wd},
              {"params": [B for _, B, _ in factors.values()], "lr": a.lr * a.b_lr_mult, "weight_decay": a.wd},
              {"params": [params[n] for n in norms], "lr": a.norm_lr, "weight_decay": 0.0}]
    groups = [grp for grp in groups if grp["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, a.beta2), fused=prepare.DEVICE.startswith("cuda"))
    trainable = {"full": norms, "adapters": ranks}
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"]
    spent = prepare.train_flops(rec, trainable, steps)["flops"]
    n_steps = int((a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - spent) // per_step)
    # mid-share, so a rounded step never puts two encodes in one harness window and none in the next
    marks = {round(n_steps * (k + 0.5) / (a.snaps + 1)) for k in range(a.snaps + 1)}
    hooks.plan(n_steps)
    print(json.dumps({"class": KLASS, "steps": n_steps, "rank": rank, "ranks": sorted(set(ranks.values())),
                      "targets": len(ranks), "norms": len(norms), "E_bytes": size, "flops_per_step": per_step,
                      "init_flops": spent, "windows": len(data.order)}), flush=True)
    t0 = time.perf_counter()
    lrs = [grp["lr"] for grp in opt.param_groups]
    trained = [p for grp in opt.param_groups for p in grp["params"]]
    ema = None
    for step in range(n_steps):
        frac = lr_factor(step, n_steps, a)
        for grp, lr in zip(opt.param_groups, lrs):
            grp["lr"] = lr * frac
        rows = data.next(a.batch)
        total = 0.0
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / a.batch)
            loss.backward()
            total += float(loss.detach())
        steps.append({"op": "train", "tokens": tokens})
        torch.nn.utils.clip_grad_norm_(trained, 1.0)
        if a.opt == "riemann":
            steps.append({"op": "extra", "flops": riemann_precondition(factors, a.riemann_delta)})
        opt.step()
        opt.zero_grad(set_to_none=True)
        ema = total if ema is None else 0.98 * ema + 0.02 * total
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": round(total, 4), "ema": round(ema, 4),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if step + 1 in marks:
            snaps.append((encode(factors, params, norms, a.dtype, a.bits), len(steps)))
    blob = encode(factors, params, norms, a.dtype, a.bits)
    model = prepare.decode(theta, blob)
    prepare.submit(Path(a.out), a.start, KLASS, blob, model, trainable, steps, changed=list(ranks) + norms,
                   notes={"method": "lora", "direction": "lowrank", "rank": rank, "ranks": ranks, "args": vars(a),
                          "steps": n_steps, "final_train_ema": ema}, snapshots=snaps)
    return 0


def encode(factors: dict, params: dict, norms: list[str], dtype: str, bits: int = 0) -> bytes:
    if bits:
        return prepare.encode_lora2({n: (A.detach().float(), B.detach().float(), s) for n, (A, B, s) in factors.items()},
                                    bits, dtype, full={n: params[n] for n in norms})
    out = {}
    for n, (A, B, s) in factors.items():
        A, B = A.detach().float(), B.detach().float()
        if dtype == "f16":
            ma, mb = float(A.abs().max()), float(B.abs().max())
            if ma > 0 and mb > 0:
                A, B, s = A / ma, B / mb, s * ma * mb
        out[n] = (A, B, s)
    return prepare.encode_lora(out, dtype, {n: params[n] for n in norms})


if __name__ == "__main__":
    raise SystemExit(main())
