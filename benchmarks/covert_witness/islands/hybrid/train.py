"""Direction `hybrid` (any class): one script that composes the other directions' mechanisms, so a pair is a flag change.

- `--full SPEC`: tensors trained in full, by AdamW or Muon (`--opt`); `--lora SPEC`: rank-r adapters (`--rank`, 0 the largest
  that fits the class's cap), AdamW with LoRA+'s faster B (`--lora-b-mult`); `--norms`: norm gains above the lowest updated
  matrix (0-FLOP nodes, so φ is unchanged) or none; `--final-norm`.
- `--store bf16` writes every full tensor through the `lora` format's bf16 `full` list: the evaluator multiplies weights at
  bf16 under autocast, so the f32 low halves `encode_full` would carry never reach a matmul. `--store f32` is `encode_full`.
  `--store qN` is CWQUAN1: each full matrix as a delta from θ in round-to-nearest N-bit codes, the adapters' factors at
  LORA_BITS, the gains in full (low-rank plus low-bit, so a deep stage fits e1mb).
- `--select-pool P`: draw P·batch windows, score them with a forward (logged), train on the `--select-by` share.
- `--cache-windows W --cache-epochs E` (the top block's MLP only): each data step reads W windows, runs θ once to the top
  MLP's input, and trains E shuffled passes over that cache, so a pass costs the top MLP and the head, not the prefix.
- `--head sampled`: the loss's softmax runs over each micro-batch's distinct targets plus `--head-neg` uniform negatives
  (log-Q corrected), so a shallow step skips most of the tied head's forward and input gradient (2·50304·d MACs a token).
- `--arms`: several settings in one candidate, each from θ with an equal share of the budget, either LR multipliers
  (`0.3,1,3`) or `;`-separated arms of `KEY=VALUE` overrides (`head=full;head=sampled;head=sampled,cache_windows=640`, keys
  in `ARM_KEYS`; `share=0.1` gives an arm that fraction of the budget, the rest split evenly). The trained set is the same
  in every arm. A later arm's checkpoints carry the arms before it, so a short complete schedule goes first.

SPEC is comma-separated `BLOCKS:LEAVES`: BLOCKS `all`, `i`, `i-j` or a negative index from the top; LEAVES `+`-joined leaf
names (`wq wk wv wo w_gate w_up w_down`) or groups (`qkv`, `attn`, `gu`, `mlp`, `all`). `--klass` must equal the class the
launcher runs (on this branch `islands/<class>/train.py` links here).

    python launch.py shallow s100 --note "..." -- --klass shallow --full=-1:gu --lr 1e-3 --arms 0.3,1,3
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
from model import rope

LEAVES = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")
GROUPS = {"qkv": ("wq", "wk", "wv"), "attn": ("wq", "wk", "wv", "wo"), "gu": ("w_gate", "w_up"),
          "mlp": ("w_gate", "w_up", "w_down"), "all": LEAVES}
ARM_KEYS = {"lr_mult": float, "lr": float, "lora_lr": float, "norm_lr": float, "lora_b_mult": float, "wd": float,
            "opt": ("adamw", "muon"), "schedule": ("linear", "cosine", "wsd"), "warmup": int, "cooldown": float,
            "batch": int, "micro": int, "select_pool": int, "select_by": ("high", "low", "mid"), "fresh": bool,
            "head": ("full", "sampled"), "head_neg": int, "head_plan": int, "cache_windows": int, "cache_epochs": int,
            "data_seed": int, "seed": int, "share": float, "snaps": int}
#: te_core takes the first E encoded in each 5% of the run's MAC budget as a checkpoint
CHECKPOINTS = 20
#: `--store qN`: entries per f16 scale, and the adapters' factor bits
QGROUP, LORA_BITS = 128, 8


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


def parse_arms(spec: str, a: argparse.Namespace) -> list[argparse.Namespace]:
    if not spec:
        items = [{}]
    elif "=" not in spec:
        items = [{"lr_mult": float(x)} for x in spec.split(",")]
    else:
        items = []
        for arm in filter(None, (s.strip() for s in spec.split(";"))):
            kv = {}
            for pair in filter(None, arm.split(",")):
                key, _, val = pair.partition("=")
                key, val = key.strip().replace("-", "_"), val.strip()
                kind = ARM_KEYS.get(key)
                if kind is None:
                    raise SystemExit(f"hybrid: --arms key {key!r} is not one of {sorted(ARM_KEYS)}")
                if isinstance(kind, tuple) and val not in kind:
                    raise SystemExit(f"hybrid: --arms {key}={val} is not one of {kind}")
                kv[key] = (val if isinstance(kind, tuple) else val.lower() in ("1", "true", "yes") if kind is bool
                           else kind(val))
            items.append(kv)
    return [argparse.Namespace(**{**vars(a), "lr_mult": 1.0, **kv}) for kv in items]


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


def rtn(t: torch.Tensor, bits: int):
    """t as round-to-nearest `bits`-bit codes into the integer codebook -2^(b-1)..2^(b-1)-1, an f16 scale per QGROUP
    entries putting each group's max |t| at the top code."""
    group = math.gcd(t.numel(), QGROUP)
    flat = t.detach().float().reshape(-1, group)
    top = 2 ** (bits - 1) - 1
    scale = (flat.abs().amax(1) / top).clamp_min(2 ** -24).half()
    codes = (flat / scale.float()[:, None]).round().clamp(-top - 1, top) + top + 1
    return prepare.QArray(torch.arange(2 ** bits, dtype=torch.float16) - (top + 1), scale.cpu(),
                          codes.to(torch.uint8).reshape(-1).cpu(), bits, group, tuple(t.shape))


def encode(store: str, factors: dict, params: dict, full: list[str], theta: dict) -> bytes:
    if store == "f32":
        if factors:
            raise SystemExit("hybrid: --store f32 is encode_full, which holds no adapters")
        return prepare.encode_full({n: params[n].detach().to(theta[n].dtype) for n in full})
    if store.startswith("q"):
        ent = {n: {"kind": "delta", "q": rtn(params[n].detach().float() - theta[n].to(params[n].device).float(),
                                             int(store[1:]))} if params[n].ndim == 2 else
               {"kind": "full", "t": params[n].detach().to(theta[n].dtype)} for n in full}
        ent |= {n: {"kind": "lora", "a": rtn(A, LORA_BITS), "b": rtn(B, LORA_BITS), "scale": s}
                for n, (A, B, s) in factors.items()}
        return prepare.encode_quant(ent, theta)
    return prepare.encode_lora({n: (A, B, s) for n, (A, B, s) in factors.items()}, "bf16", {n: params[n] for n in full})


def e_bytes(store: str, theta: dict, lora: list[str], rank: int, full: list[str]) -> int:
    """E's size at this rank; for `--store qN` an upper bound, with every code and every group's scale drawn uniformly
    (xz can't shrink them)."""
    g = torch.Generator().manual_seed(0)

    def uniform(*shape: int) -> torch.Tensor:
        n = math.prod(shape)
        k = math.gcd(n, QGROUP)
        return ((torch.rand(n // k, k, generator=g) - 0.5) * (0.5 + torch.rand(n // k, 1, generator=g))).reshape(shape)

    fill = uniform if store.startswith("q") else torch.zeros
    fac = {n: (fill(rank, theta[n].shape[1]), fill(theta[n].shape[0], rank), math.sqrt(theta[n].shape[1] / rank))
           for n in lora}
    vals = {n: theta[n].float() + fill(*theta[n].shape) if store.startswith("q") else theta[n] for n in full}
    return len(encode(store, fac, vals, full, theta))


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
    ap.add_argument("--store", choices=("bf16", "f32", *(f"q{b}" for b in range(2, 9))), default="bf16")
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
    ap.add_argument("--fresh", action=argparse.BooleanOptionalAction, default=False,
                    help="train only on windows the start never read")
    ap.add_argument("--head", choices=("full", "sampled"), default="full",
                    help="sampled: softmax over each micro-batch's distinct targets plus --head-neg uniform negatives")
    ap.add_argument("--head-neg", type=int, default=2048)
    ap.add_argument("--head-plan", type=int, default=10240,
                    help="head rows a micro-batch is planned at, for the step count (the log prints the mean measured)")
    ap.add_argument("--cache-windows", type=int, default=0,
                    help="cache θ's top-MLP input for this many windows per step and train --cache-epochs passes on it")
    ap.add_argument("--cache-epochs", type=int, default=1)
    ap.add_argument("--snaps", type=int, default=8, help="E snapshots at k/SNAPS of each arm, for the FLOPs-to-target curve")
    ap.add_argument("--arms", default="", help="LR multipliers (0.3,1,3) or ';'-separated arms of KEY=VALUE overrides, "
                    "each from θ with an equal share of the budget (later arms' checkpoints are charged the arms "
                    "before them)")
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    ap.add_argument("--share", type=float, default=0.0, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.store.startswith("q") and not hasattr(prepare, "encode_quant"):
        raise SystemExit("hybrid: --store qN needs prepare.encode_quant (CWQUAN1)")
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

    def optimizers(aa: argparse.Namespace) -> list[torch.optim.Optimizer]:
        m = aa.lr_mult
        groups = []
        if norms:
            groups.append({"params": [params[n] for n in norms], "lr": aa.norm_lr * m, "weight_decay": 0.0})
        if factors:
            groups.append({"params": [A for A, _, _ in factors.values()], "lr": aa.lora_lr * m, "weight_decay": 0.0})
            groups.append({"params": [B for _, B, _ in factors.values()], "lr": aa.lora_lr * aa.lora_b_mult * m,
                           "weight_decay": 0.0})
        if full_m and aa.opt == "adamw":
            groups.append({"params": [params[n] for n in full_m], "lr": aa.lr * m, "weight_decay": aa.wd})
        opts = [torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=fused)] if groups else []
        if full_m and aa.opt == "muon":
            opts.append(Muon([params[n] for n in full_m], lr=aa.lr * m, wd=aa.wd))
        return opts

    @torch.no_grad()
    def reset(seed: int) -> None:
        for n in full:
            params[n].copy_(theta[n].to(params[n].device, torch.float32))
        g.manual_seed(seed)
        for A, B, _ in factors.values():
            A.copy_(torch.randn(A.shape, device=prepare.DEVICE, generator=g) / math.sqrt(A.shape[1]))
            B.zero_()

    trainable = {"full": full, "adapters": {n: rank for n in lora}}
    V, d = net.inner.embed.shape
    arms = parse_arms(a.arms, a)
    fixed, free = sum(aa.share for aa in arms), [aa for aa in arms if not aa.share]
    if fixed > 1 + 1e-9 or (free and fixed >= 1) or any(aa.share < 0 for aa in arms):
        raise SystemExit(f"hybrid: arm shares {[aa.share for aa in arms]} don't fit one budget")
    for aa in free:
        aa.share = (1 - fixed) / len(free)
    cached = [aa for aa in arms if aa.cache_windows]
    if cached:
        top = {f"blocks.{L - 1}.{x}.weight" for x in ("w_gate", "w_up", "w_down")}
        if set(full_m + lora) - top or any(aa.select_pool > 1 or aa.cache_windows % aa.batch for aa in cached):
            raise SystemExit("hybrid: --cache-windows trains only the top block's MLP (and the gains above it), without "
                             "selection, on a multiple of --batch windows")
        nodes = prepare.positions(rec)
        fwd_tok = prepare.train_flops(rec, trainable, [{"op": "forward", "tokens": 1}])["flops"]
        prefix_tok = fwd_tok - sum(n["flops"] for n in nodes if (n["layer"], n["stage"]) >= (L, 4))

    def plan(aa: argparse.Namespace) -> dict:
        """An arm's steps within its share: n_data data steps (TrainData.next calls), n_train optimizer steps."""
        share = aa.share * a.budget_fraction * prepare.TRAIN_FLOP_BUDGET
        tokens = aa.batch * prepare.EVAL_CTX
        extra = sum(ns_flops(tuple(params[n].shape)) for n in full_m) if aa.opt == "muon" else 0
        pool = aa.select_pool * aa.batch
        per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}] + (
            [{"op": "forward", "tokens": pool * prepare.EVAL_CTX}] if aa.select_pool > 1 else []))["flops"]
        if aa.head == "sampled":
            per_step -= 2 * 2 * (V - aa.head_plan) * d * tokens
        p = {"tokens": tokens, "extra": extra, "pool": pool, "flops_per_step": per_step + extra, "planned_flops": share}
        W, E = aa.cache_windows, aa.cache_epochs
        if W:
            epoch_tok = per_step / tokens - prefix_tok
            chunk = W * prepare.EVAL_CTX * (prefix_tok + E * epoch_tok) + E * (W // aa.batch) * extra
            n_chunks = int(share // chunk)
            p.update(epoch_tok=epoch_tok, flops_per_chunk=chunk, n_data=n_chunks, n_train=n_chunks * E * (W // aa.batch),
                     reads=n_chunks * W)
        else:
            n = int(share // (per_step + extra))
            p.update(n_data=n, n_train=n, reads=n * pool)
        return p

    plans = [plan(aa) for aa in arms]
    fwd = torch.compile(net) if fused else net
    dev = prepare.DEVICE.split(":")[0]
    structure = prepare.deep_share(rec, set(full + lora))
    steps, snaps, arms_out, cands = [], [], [], []
    gneg = torch.Generator(device=prepare.DEVICE)
    t0 = time.perf_counter()
    for arm, (aa, p) in enumerate(zip(arms, plans)):
        reset(aa.seed)
        opts = optimizers(aa)
        gneg.manual_seed(aa.seed + 1)
        base_lrs = [[grp["lr"] for grp in o.param_groups] for o in opts]
        data = prepare.TrainData(rec["tokenizer"], seed=aa.data_seed)
        fresh = fresh_order(data, rec) if aa.fresh else {"fresh": False}
        if aa.fresh and fresh.get("kept", 0) < p["reads"]:
            raise SystemExit(f"hybrid: --fresh leaves {fresh.get('kept')} windows, the arm reads {p['reads']}")
        n_data, n_train, last = p["n_data"], p["n_train"], arm == len(arms) - 1
        # a non-last arm's end must be the first E encoded in its 5%, so no snapshot within a 5% (10% slack) of it
        keep = 1.0 if last else 1 - 1.1 / (CHECKPOINTS * aa.share)
        marks = {round(n_data * j / aa.snaps) for j in range(1, aa.snaps) if j / aa.snaps < keep} - {0}
        if not last:
            marks.add(n_data)
        overrides = {k: v for k, v in vars(aa).items() if k not in vars(a) or v != vars(a)[k]}
        arms_out.append({**overrides, "first_train_step": sum(q["n_train"] for q in plans[:arm]), "train_steps": n_train,
                         "data_steps": n_data, "planned_flops": p["planned_flops"]})
        print(json.dumps({"arm": arm, **overrides, "train_steps": n_train, "data_steps": n_data, "full": len(full_m),
                          "lora": len(lora), "rank": rank if lora else None, "norms": len(norms), "E_bytes": size,
                          "flops_per_step": p["flops_per_step"], "phi": structure["phi"],
                          "lowest_updated_layer": structure["lowest_updated_layer"], "data": fresh}), flush=True)
        k = 0

        def train_step(micro_losses) -> None:
            nonlocal k
            frac = lr_frac(k, n_train, aa.warmup, aa.schedule, aa.cooldown)
            for o, lrs in zip(opts, base_lrs):
                for grp, lr in zip(o.param_groups, lrs):
                    grp["lr"] = lr * frac
            total = 0.0
            for fn, part in micro_losses:
                with torch.autocast(dev, dtype=torch.bfloat16):
                    loss = fn()
                (loss * part).backward()
                total += float(loss.detach()) * part
            torch.nn.utils.clip_grad_norm_([q for o in opts for grp in o.param_groups for q in grp["params"]], 1.0)
            for o in opts:
                o.step()
                o.zero_grad(set_to_none=True)
            if p["extra"]:
                steps.append({"op": "extra", "flops": p["extra"], "what": "muon newton-schulz"})
            k += 1
            if k % 100 == 0:
                print(json.dumps({"arm": arm, "step": k, "loss": round(total, 5), "s": round(time.perf_counter() - t0, 1),
                                  **({"candidates": round(sum(cands) / len(cands))} if cands else {})}), flush=True)

        def head(hn, tgt):
            if aa.head == "full":
                return F.cross_entropy(F.linear(hn, net.inner.embed).float().view(-1, V), tgt.reshape(-1))
            loss, n_cand = sampled_loss(net.inner, hn, tgt, aa.head_neg, gneg)
            cands.append(n_cand)
            return loss

        W, E, B, mb = aa.cache_windows, aa.cache_epochs, aa.batch, aa.micro
        gshuf = torch.Generator(device="cpu").manual_seed(aa.seed + 2)
        for c in range(1, n_data + 1):
            if W:
                rows = data.next(W)
                with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
                    mids = torch.cat([after_attention(net.inner, rows[i: i + mb, :-1]) for i in range(0, W, mb)])
                steps.append({"op": "forward", "tokens": W * prepare.EVAL_CTX, "what": "cache: theta to the top MLP's input"})
                for _ in range(E):
                    perm = torch.randperm(W, generator=gshuf).to(rows.device)
                    for b in range(0, W, B):
                        idx = perm[b: b + B]
                        train_step([(lambda j=idx[i: i + mb]: head(top_mlp(net.inner, mids[j]), rows[j, 1:]),
                                     len(idx[i: i + mb]) / B) for i in range(0, B, mb)])
                        steps.append({"op": "extra", "flops": int(B * prepare.EVAL_CTX * p["epoch_tok"]),
                                      "what": "cached top-MLP step"})
            else:
                rows = data.next(p["pool"])
                if aa.select_pool > 1:
                    rows = choose(fwd, rows, B, mb, aa.select_by, dev)
                    steps.append({"op": "forward", "tokens": p["pool"] * prepare.EVAL_CTX,
                                  "what": f"selection by {aa.select_by} loss"})
                if aa.head == "full":
                    fns = [lambda x=rows[i: i + mb]: F.cross_entropy(
                        fwd(x[:, :-1]).float().view(-1, V), x[:, 1:].reshape(-1)) for i in range(0, B, mb)]
                else:
                    fns = [lambda x=rows[i: i + mb]: head(F.rms_norm(net.inner.hidden(x[:, :-1]), (d,), net.inner.norm),
                                                          x[:, 1:]) for i in range(0, B, mb)]
                train_step([(fn, len(rows[i: i + mb]) / B) for fn, i in zip(fns, range(0, B, mb))])
                steps.append({"op": "train", "tokens": p["tokens"]})
            if c in marks:
                snaps.append((encode(a.store, factors, params, full, theta), len(steps)))
    blob = encode(a.store, factors, params, full, theta)
    model = prepare.decode(theta, blob)
    prepare.submit(Path(a.out), a.start, a.klass, blob, model, trainable, steps, changed=full + lora,
                   notes={"method": "hybrid", "rank": rank if lora else None, "args": vars(a),
                          "steps": sum(q["n_train"] for q in plans), "arms": arms_out, "data": fresh,
                          **({"head_candidates_mean": sum(cands) / len(cands)} if cands else {})},
                   snapshots=snaps)
    return 0


def sampled_loss(lm, hn: torch.Tensor, tgt: torch.Tensor, k: int, gen: torch.Generator) -> tuple[torch.Tensor, int]:
    """Cross-entropy through the tied head over the micro-batch's distinct targets plus k negatives drawn uniformly from the
    rest of the vocabulary, each carrying log((V - |targets|) / k) for the rest's share of the partition function (sampled
    softmax with its log-Q correction). The head rows are gathered with index_select, so the token count stays the batch's."""
    cand, idx = torch.unique(tgt, return_inverse=True)
    z = F.linear(hn, lm.embed.index_select(0, cand))
    V = lm.embed.shape[0]
    if k and V - len(cand) > k:
        rest = torch.ones(V, device=hn.device)
        rest[cand] = 0.0
        neg = torch.multinomial(rest, k, replacement=False, generator=gen)
        z = torch.cat([z, F.linear(hn, lm.embed.index_select(0, neg)) + math.log((V - len(cand)) / k)], -1)
    return F.cross_entropy(z.float().view(-1, z.shape[-1]), idx.reshape(-1)), len(cand)


def after_attention(lm, inp: torch.Tensor) -> torch.Tensor:
    """θ's residual stream after the top block's attention: the top MLP's input, cached for --cache-epochs passes."""
    blk = lm.blocks[-1]
    x = lm.hidden(inp, len(lm.blocks) - 1)
    b, t, d = x.shape
    cos, sin = lm.cos[:t], lm.sin[:t]
    h = F.rms_norm(x, (d,), blk.attn_norm)
    q, k, v = (w(h).view(b, t, blk.n_head, d // blk.n_head).transpose(1, 2) for w in (blk.wq, blk.wk, blk.wv))
    q, k = rope(q, cos, sin), rope(k, cos, sin)
    return x + blk.wo(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d))


def top_mlp(lm, x: torch.Tensor) -> torch.Tensor:
    """The top block's MLP and the final norm on a cached residual stream."""
    blk = lm.blocks[-1]
    d = x.shape[-1]
    h = F.rms_norm(x, (d,), blk.mlp_norm)
    return F.rms_norm(x + blk.w_down(F.silu(blk.w_gate(h)) * blk.w_up(h)), (d,), lm.norm)


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
