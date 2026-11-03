"""Island `shallow`: updates confined to the top blocks and the final norm, so the frozen layers below are computed from the
token's data. On a pin with classes (v2.3), `--class shallow` is the harness's cap (the top 2 blocks) and `--class
reference` takes any set; v3's prepare has none, and submit gets no class there. The rest stays frozen and the backward
stops at the lowest tuned node. The head is the tied embedding, which the input lookup also reads (layer 0), so it stays
frozen.

What is tuned: tensors trained in full (`--full`) and matrices trained through rank-`--lora` factors (`--adapt`), each a
comma list of `tN.LEAF` (block N from the top, t0 the last; LEAF one of LF_LEAVES or `*`) and `final` (the final norm).
Without them, `--blocks`/`--parts`/`--final` name the full set. The first updated stage has no updated multiplication
upstream, so it is never deep: `--full t0.w_gate,t0.w_up,final` has φ = 0 and depth 0, and `--full t1.w_gate,t1.w_up
--adapt t1.w_down,t0.* --lora 4` keeps every later update low-rank.

How: AdamW or Muon (`--opt`) on full matrices, AdamW on gains and factors. E (`--store`): `fp32` every changed tensor
in θ's dtype, adapters merged (`prepare.encode_full`); `bf16` factors and full tensors in bf16 (the lora format; the
bf16-autocast forward already multiplies at bf16); `svd:R` each full matrix's delta truncated to rank R as bf16 factors;
`qB` or `qB:G` each tensor's delta from θ as a CWQUAN1 delta (`prepare.encode_quant`): B-bit round-to-nearest codes into
a symmetric codebook, an f16 scale per G entries (default 128), xz-coded; `uB` or `uB:S` the same format on a uniform
step of S (default 1) times each 128 entries' RMS, zero a level, so xz's entropy coding sets the rate (`uniform_code`),
and `--max-e-bytes N` fits the step so every checkpoint's |E| is just under N (or under it at the finest step whose
levels span 6 RMS, `UNIFORM_COVER`); `auto` is bf16 with adapters, else
fp32. An
arm's `store=` codes its own E (`svd:R` only as the global store). `--mom-dtype bf16` keeps Muon's momentum in bf16
between steps and `--grad-dtype bf16` rounds the accumulated gradient to bf16 after each micro-batch; `--mom-fold` makes
Muon heavy-ball momentum with each micro-batch's gradient summed straight into that buffer. Each arm's report
records |E|, its trainable parameters and the state training carries (`v1`: the gradient, the optimizer's state, the fp32
weights), the submitted arm's Δ_grad goes in `notes.delta_grad`, and, for `qB`, the code's relative error on the delta. `--fresh` trains only on windows the start never read (its provenance
names its order: default_rng(data_seed), step x batch). `--ema D` submits an exponential average of the full tensors.
The submitted model is always decode(θ, E).

Checkpoints: the harness scores the first E encoded in each 5% of the planned steps (`hooks.plan`), so E is encoded at
the first step of each 5%, and a single arm's end is the submission's checkpoint. `--arms "lr=1e-4;lr=3e-4"` splits the budget into runs from θ in turn (a sweep in one
candidate), in proportion to each arm's `share` (default 1; `share=1;share=2;share=4` sweeps run length, each arm's end
fully decayed): no other encode shares an arm's last 5%, so every arm's end is a checkpoint; later arms' checkpoints carry
the earlier arms' MACs; E is the arm whose last tenth of batches (each unseen when its loss is taken) had the lowest loss.
An arm's `full=SPEC` (no spaces) trains its own full set, every arm starting from θ on the union: order such arms
shallow to deep with nested sets (`full=t0.w_gate,t0.w_up,final;full=t0.*,final`), since a checkpoint's φ is read from
every update through it.

This file is the island's one editable file; the fixed side is README.md's list. candidate.sh runs it inside the harness:

    train.py --start m100 --out SUB [--class shallow] [--full SPEC] [--adapt SPEC --lora R] [--opt adamw]
        [--store q8] [--arms "lr=1e-4;lr=3e-4"]
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import re
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import prepare

try:
    from te_core import hooks
except ImportError:
    hooks = None

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
    """Nesterov momentum, orthogonalized. The momentum buffer is kept in `buf_dtype` between steps (bf16: summed in
    fp32, rounded to nearest on store, and the step reads the rounded buffer).

    `fold=True` is heavy-ball momentum whose buffer is also the gradient's accumulator: `decay()` scales it by the
    momentum before a step's first backward, `fold()` adds each micro-batch's gradient into it (rounded to `buf_dtype`)
    and drops the gradient, and the step orthogonalizes the buffer alone, so no gradient outlives a micro-batch."""

    def __init__(self, params, lr: float, momentum: float = 0.95, wd: float = 0.0, buf_dtype=torch.float32,
                 fold: bool = False):
        super().__init__(params, {"lr": lr, "momentum": momentum, "wd": wd})
        self.buf_dtype, self.fold_grads = buf_dtype, fold

    def buf(self, p: torch.Tensor) -> torch.Tensor:
        st = self.state[p]
        if "buf" not in st:
            st["buf"] = torch.zeros_like(p, dtype=self.buf_dtype)
        return st["buf"]

    @torch.no_grad()
    def decay(self):
        for g in self.param_groups:
            for p in g["params"]:
                self.buf(p).mul_(g["momentum"])

    @torch.no_grad()
    def fold(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    b = self.buf(p)
                    b.copy_(b.float().add_(p.grad))
                    p.grad = None

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if self.fold_grads:
                    u = newton_schulz(self.buf(p).float())
                elif p.grad is None:
                    continue
                else:
                    b = self.buf(p)
                    buf = b.copy_(b.float().mul_(g["momentum"]).add_(p.grad)).float()
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


def seen_windows(rec: dict, n_seq: int) -> np.ndarray | None:
    """The windows the start read (its provenance: default_rng(data_seed), step x batch), or None if it names no order."""
    prov = rec.get("provenance", {})
    if not {"step", "batch", "data_seed"} <= set(prov):
        return None
    return np.random.default_rng(prov["data_seed"]).permutation(n_seq)[: prov["step"] * prov["batch"]]


def fresh_order(data, rec: dict) -> dict:
    """Drop the windows the start read from the stream's order."""
    n_seq = (len(data.tokens) - 1) // data.ctx
    seen = seen_windows(rec, n_seq)
    if seen is None:
        return {"fresh": False, "why": "the start's provenance names no data order"}
    keep = np.ones(n_seq, dtype=bool)
    keep[seen] = False
    before = len(data.order)
    data.order = data.order[keep[data.order]]
    return {"fresh": True, "windows": n_seq, "seen": int(len(seen)), "kept": int(len(data.order)), "of": before}


#: the harness scores the first E encoded in each 1/CHECKPOINTS of the planned steps
CHECKPOINTS = 20
ARM_KEYS = {"lr": float, "muon_lr": float, "lora_lr": float, "opt": str, "wd": float, "ema": float, "schedule": str,
            "warmup": int, "batch": int, "micro": int, "fresh": lambda v: v.lower() in ("1", "true", "on", "yes"),
            "data_seed": int, "seed": int, "share": float, "full": str, "store": str, "mom_dtype": str,
            "grad_dtype": str, "mom_fold": lambda v: v.lower() in ("1", "true", "on", "yes")}
ARM_CHOICES = {"opt": ("adamw", "muon"), "schedule": ("linear", "cosine", "wsd"), "mom_dtype": ("fp32", "bf16"),
               "grad_dtype": ("fp32", "bf16")}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
BITS = {"fp32": 32, "bf16": 16}


def parse_arms(a: argparse.Namespace) -> list[argparse.Namespace]:
    """`--arms "lr=1e-4;lr=3e-4 wd=0.1"`: runs from θ in turn, each overriding how it trains (ARM_KEYS) and, with
    `full=SPEC`, which tensors it trains in full."""
    a.share = 1.0
    if not a.arms:
        return [a]
    arms = []
    for text in filter(None, (t.strip() for t in a.arms.split(";"))):
        arm = argparse.Namespace(**vars(a))
        arm.overrides = text
        for kv in text.split():
            k, _, v = kv.partition("=")
            k = k.replace("-", "_")
            if k not in ARM_KEYS or not v:
                raise SystemExit(f"shallow: arm item {kv!r} is not KEY=VALUE with KEY in {sorted(ARM_KEYS)}")
            setattr(arm, k, ARM_KEYS[k](v))
            if k in ARM_CHOICES and getattr(arm, k) not in ARM_CHOICES[k]:
                raise SystemExit(f"shallow: arm {k}={v} is not one of {ARM_CHOICES[k]}")
        arms.append(arm)
    return arms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--class", dest="klass", choices=("shallow", "reference"), default="shallow",
                    help="shallow: the top 2 blocks at most; reference: any set (a pin whose prepare has CLASSES)")
    ap.add_argument("--full", default=None, help="tensors trained in full: tN.LEAF, tN.*, final (default: --blocks/--parts)")
    ap.add_argument("--adapt", default="", help="matrices trained through rank --lora factors: tN.LEAF, tN.*")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--parts", default="all", help="`all` or leaf names in each top block: " + ",".join(LF_LEAVES))
    ap.add_argument("--final", action=argparse.BooleanOptionalAction, default=True, help="tune the final norm")
    ap.add_argument("--opt", choices=("adamw", "muon"), default="adamw")
    ap.add_argument("--lr", type=float, default=3e-4, help="AdamW's rate (gains, and full matrices unless --opt muon)")
    ap.add_argument("--muon-lr", type=float, default=0.01)
    ap.add_argument("--mom-dtype", choices=("fp32", "bf16"), default="fp32", help="Muon's momentum buffer between steps")
    ap.add_argument("--grad-dtype", choices=("fp32", "bf16"), default="fp32",
                    help="the gradient as accumulated and passed (bf16: rounded after every micro-batch's backward)")
    ap.add_argument("--mom-fold", action=argparse.BooleanOptionalAction, default=False,
                    help="Muon as heavy-ball momentum with the gradient summed straight into its --mom-dtype buffer")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lora", type=int, default=0, help="rank of the --adapt factors (alone: every tuned matrix)")
    ap.add_argument("--lora-lr", type=float, default=1e-3)
    ap.add_argument("--store", default="auto", help="auto | fp32 | bf16 | svd:R | qB[:G] | uB[:S]")
    ap.add_argument("--max-e-bytes", type=int, default=None, help="uB: widen the step until |E| fits")
    ap.add_argument("--fresh", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--schedule", choices=("linear", "cosine", "wsd"), default="linear")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-fraction", type=float, default=0.95, help="of TRAIN_FLOP_BUDGET by the analytic meter")
    ap.add_argument("--arms", default="", help="`;`-separated runs from θ sharing the budget, each `KEY=VALUE ...` over "
                                               + ",".join(ARM_KEYS) + "; each arm's end is a checkpoint")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()
    arms = parse_arms(a)
    torch.backends.cuda.matmul.allow_tf32 = True
    if prepare.start_record(a.start)["config"]["family"] != "lf":
        raise SystemExit("shallow: this train.py tunes the lf starts (s50, s100, m50, m100)")
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    default = a.full if a.full is not None else default_full(a.blocks, a.parts, a.final)
    adapt = [n for n in spec_names(rec, params, a.adapt) if params[n].ndim == 2]
    for arm in arms:
        arm.tuned = spec_names(rec, params, arm.full if arm.full is not None else default)
        if a.lora and not a.adapt:
            if arm.full != a.full:
                raise SystemExit("shallow: an arm's full= needs --adapt with --lora")
            adapt = [n for n in arm.tuned if params[n].ndim == 2]
            arm.tuned = [n for n in arm.tuned if params[n].ndim == 1]
        if set(arm.tuned) & set(adapt):
            raise SystemExit("shallow: --adapt names matrices that --full does not")
        arm.mats = [n for n in arm.tuned if params[n].ndim == 2]
        arm.gains = [n for n in arm.tuned if params[n].ndim == 1]
        if arm.mom_fold and arm.opt != "muon":
            raise SystemExit("shallow: --mom-fold is Muon's (--opt muon)")
    if adapt and not a.lora:
        raise SystemExit("shallow: --adapt needs --lora R")
    full = sorted({n for arm in arms for n in arm.tuned})
    store = a.store if a.store != "auto" else ("bf16" if adapt else "fp32")
    for arm in arms:
        arm.store = arm.store if arm.store != "auto" else store
        if not re.fullmatch(r"fp32|bf16|svd:\d+|q[1-8](:\d+)?|u[2-8](:[0-9.]+)?", arm.store):
            raise SystemExit(f"shallow: --store {arm.store} is not fp32, bf16, svd:R, qB[:G] (B in 1..8) or uB[:S] (B in 2..8)")
        if arm.store.startswith("svd:") != store.startswith("svd:") or (arm.store[0] in "qu" and adapt):
            raise SystemExit("shallow: svd:R is the global --store only, and qB and uB code full tensors' deltas (no --adapt)")
        if a.max_e_bytes is not None and arm.store[0] != "u":
            raise SystemExit("shallow: --max-e-bytes fits a uB store's step")
    for p in params.values():
        p.requires_grad_(False)
    for n in full:
        if params[n].dtype != torch.float32:
            params[n].data = params[n].data.float()
        params[n].requires_grad_(True)

    factors = {}
    for n in adapt:
        w = params[n]
        A = torch.nn.Parameter(torch.empty(a.lora, w.shape[1], device=prepare.DEVICE))
        B = torch.nn.Parameter(torch.zeros(w.shape[0], a.lora, device=prepare.DEVICE))
        s = math.sqrt(w.shape[1] / a.lora)
        factors[n] = (A, B, s)
        mod = net.inner.get_submodule(n.removesuffix(".weight"))
        mod.register_forward_hook(lambda m, inp, out, A=A, B=B, s=s: out + s * F.linear(F.linear(inp[0], A), B))

    trainable = {"full": full, "adapters": {n: a.lora for n in adapt}}
    mats = [n for n in full if params[n].ndim == 2]
    store_extra = svd_flops(params, mats) if store.startswith("svd:") else 0
    c = prepare.EVAL_CTX
    budget = a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - (CHECKPOINTS + len(arms)) * store_extra
    plans = []
    for arm in arms:
        extra = sum(ns_flops(tuple(params[n].shape)) for n in arm.mats) if arm.mats and arm.opt == "muon" else 0
        own = {"full": arm.tuned, "adapters": trainable["adapters"]}
        per_step = prepare.train_flops(rec, own, [{"op": "train", "tokens": arm.batch * c}])["flops"] + extra
        plans.append((per_step, extra, int(budget * arm.share / sum(x.share for x in arms) // per_step)))
    total = sum(n for _, _, n in plans)
    if hooks is not None:
        hooks.plan(total)

    def bucket(g: int) -> int:
        return math.ceil((g + 1) * CHECKPOINTS / total)

    ends = np.cumsum([n for _, _, n in plans]) - 1
    # one arm's end is the submission, always a checkpoint, so its last 5% also gets the E at its first step
    end_buckets = {bucket(int(e)) for e in ends} if len(arms) > 1 else set()
    fwd = torch.compile(net) if a.compile else net
    fused = prepare.DEVICE.startswith("cuda")
    steps, report, blobs, encoded = [], [], [], set()
    g = -1
    t0 = time.perf_counter()

    for k, arm in enumerate(arms):
        mats, gains = arm.mats, arm.gains
        with torch.no_grad():
            for n in full:
                params[n].copy_(theta[n].to(params[n].device, torch.float32))
                params[n].requires_grad_(n in arm.tuned)
            gen = torch.Generator(device=prepare.DEVICE).manual_seed(arm.seed)
            for A, B, _ in factors.values():
                A.copy_(torch.randn(A.shape, device=prepare.DEVICE, generator=gen) / math.sqrt(A.shape[1]))
                B.zero_()
        per_step, extra, n_steps = plans[k]
        opts = []
        if factors:
            opts.append((torch.optim.AdamW([p for A, B, _ in factors.values() for p in (A, B)], lr=arm.lora_lr,
                                           betas=(0.9, 0.95), weight_decay=0.0, fused=fused), arm.lora_lr))
        if mats and arm.opt == "muon":
            opts.append((Muon([params[n] for n in mats], lr=arm.muon_lr, wd=arm.wd,
                              buf_dtype=DTYPES[arm.mom_dtype], fold=arm.mom_fold), arm.muon_lr))
        elif mats:
            opts.append((torch.optim.AdamW([params[n] for n in mats], lr=arm.lr, betas=(0.9, 0.95), weight_decay=arm.wd,
                                           fused=fused), arm.lr))
        if gains:
            opts.append((torch.optim.AdamW([params[n] for n in gains], lr=arm.lr, betas=(0.9, 0.95), weight_decay=0.0,
                                           fused=fused), arm.lr))
        trained = [p for o, _ in opts for grp in o.param_groups for p in grp["params"]]
        folder = next((o for o, _ in opts if isinstance(o, Muon) and o.fold_grads), None)
        tokens = arm.batch * c
        data = prepare.TrainData(rec["tokenizer"], seed=arm.data_seed)
        fresh = fresh_order(data, rec) if arm.fresh else {"fresh": False}
        ema = {n: params[n].detach().float().clone() for n in arm.tuned} if arm.ema else None
        tail_from = n_steps - max(1, n_steps // 10)
        tail = torch.zeros((), device=prepare.DEVICE)
        print(json.dumps({"arm": k, "overrides": getattr(arm, "overrides", ""), "steps": n_steps, "planned": total,
                          "full": arm.tuned, "adapt": adapt, "rank": a.lora, "store": arm.store,
                          "flops_per_step": per_step, "extra_per_step": extra, "fresh": fresh}), flush=True)
        code = {}

        def encode() -> bytes:
            cur = {n: (ema[n] if ema is not None else params[n].detach()).float() for n in arm.tuned}
            return encode_update(arm.store, theta, cur, factors, mats, gains, params, steps, code, a.max_e_bytes)

        for step in range(n_steps):
            g += 1
            frac = min(1.0, (step + 1) / arm.warmup) * decay(arm.schedule, step, n_steps)
            for o, lr in opts:
                for grp in o.param_groups:
                    grp["lr"] = lr * frac
            rows = data.next(arm.batch)
            if folder is not None:
                folder.decay()
            for i in range(0, arm.batch, arm.micro):
                x = rows[i: i + arm.micro]
                with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
                    logits = fwd(x[:, :-1])
                loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / arm.batch)
                loss.backward()
                if folder is not None:
                    folder.fold()
                if arm.grad_dtype == "bf16":
                    with torch.no_grad():
                        for p in trained:
                            if p.grad is not None:
                                p.grad.copy_(p.grad.bfloat16())
                if step >= tail_from:
                    tail += loss.detach()
            torch.nn.utils.clip_grad_norm_(trained, 1.0)
            for o, _ in opts:
                o.step()
                o.zero_grad(set_to_none=True)
            if ema is not None:
                with torch.no_grad():
                    for n in arm.tuned:
                        ema[n].lerp_(params[n].detach().float(), 1 - arm.ema)
            steps.append({"op": "train", "tokens": tokens})
            if extra:
                steps.append({"op": "extra", "flops": extra, "what": "muon newton-schulz"})
            if (step + 1) % 100 == 0:
                print(json.dumps({"arm": k, "step": step + 1, "loss": float(loss.detach()) * arm.batch / len(x),
                                  "s": round(time.perf_counter() - t0, 1)}), flush=True)
            b = bucket(g)
            if step == n_steps - 1:
                blob = encode()
                encoded.add(g)
            elif b not in end_buckets and not any(bucket(e) == b for e in encoded):
                encode()
                encoded.add(g)
        entry = {"overrides": getattr(arm, "overrides", ""), "full": arm.tuned, "steps": n_steps, "last_step": g,
                 "fresh": fresh, "tail_loss": float(tail) / (n_steps - tail_from), "store": arm.store,
                 "e_bytes": len(blob), "code": code, "v1": v1_state(arm, params, factors)}
        print(json.dumps({"arm": k, "last_step": g, "tail_loss": entry["tail_loss"], "e_bytes": len(blob), "code": code,
                          "v1": entry["v1"]}), flush=True)
        report.append(entry)
        blobs.append(blob)
    best = min(range(len(arms)), key=lambda k: report[k]["tail_loss"])
    blob = blobs[best]
    model = prepare.decode(theta, blob)
    tuned = sorted(full + adapt)
    changed = sorted(n for n in tuned if not prepare.same_bits(model[n], theta[n]))
    v1 = report[best]["v1"]
    delta_grad = {"trainable": v1["params"], **{k: v1[k] for k in ("grad_bits", "adam_bits", "muon_bits", "ema_bits",
                                                                    "latent_bits", "folded")},
                  "grad_dtype": arms[best].grad_dtype, "mom_dtype": arms[best].mom_dtype,
                  "bits": v1["delta_grad_bits"], "bytes": v1["delta_grad_bits"] // 8}
    klass = (a.klass,) if hasattr(prepare, "CLASSES") else ()
    prepare.submit(Path(a.out), a.start, *klass, blob, model, trainable, steps, changed=changed or tuned,
                   notes={"method": "shallow", "args": vars(a), "steps": report[best]["steps"], "full": full,
                          "adapt": adapt, "store": arms[best].store, "fresh": report[best]["fresh"], "arms": report,
                          "submitted_arm": best, "delta_grad": delta_grad, "planned_steps": total,
                          "encoded_steps": sorted(encoded),
                          "seconds": round(time.perf_counter() - t0, 1)})
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


def v1_state(arm: argparse.Namespace, params: dict, factors: dict) -> dict:
    """What training carries between steps, in bits (Δ_grad is the gradient plus the optimizer's state as passed): the
    gradient at --grad-dtype; AdamW's fp32 exp_avg and exp_avg_sq, or Muon's one momentum buffer per matrix at
    --mom-dtype (gains and factors on AdamW), which under --mom-fold is also the matrices' gradient; the EMA's fp32 copy
    if any. The fp32 weights are listed apart."""
    mat = sum(params[n].numel() for n in arm.mats)
    gain = sum(params[n].numel() for n in arm.gains)
    lora = sum(A.numel() + B.numel() for A, B, _ in factors.values())
    n = mat + gain + lora
    muon = arm.opt == "muon" and mat > 0
    adam = 64 * (gain + lora + (0 if muon else mat))
    muon_bits = BITS[arm.mom_dtype] * mat if muon else 0
    ema = 32 * (mat + gain) if arm.ema else 0
    grad = BITS[arm.grad_dtype] * (n - mat if muon and arm.mom_fold else n)
    state = adam + muon_bits + ema
    return {"params": n, "opt": arm.opt, "grad_bits": grad, "adam_bits": adam, "muon_bits": muon_bits, "ema_bits": ema,
            "latent_bits": 0, "folded": bool(muon and arm.mom_fold), "state_bits": state, "weights_bits": 32 * n,
            "delta_grad_bits": grad + state}


@torch.no_grad()
def merged(w: torch.Tensor, A: torch.Tensor, B: torch.Tensor, s: float) -> torch.Tensor:
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return (w.float() + s * (B.detach().float() @ A.detach().float())).to(w.dtype)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def quant_size(entries: dict, theta) -> int:
    """|E| as prepare.encode_quant writes `entries` (its header, arrays and xz), without calling it: the harness takes
    the first E encoded in each 5% as that checkpoint, so a step's trial sizes must not go through it."""
    header, parts = [], []
    for name in sorted(entries):
        spec, arrays = prepare._quant_array(entries[name]["q"], name)
        header.append({"n": name, "k": "delta", **spec})
        parts += arrays
    h = json.dumps({"t": header}, separators=(",", ":")).encode()
    return 8 + len(lzma.compress(struct.pack("<I", len(h)) + h + b"".join(parts), **prepare.QUANT_XZ))


def uniform_code(theta, cur, bits: int, step: float) -> tuple[dict, float]:
    """Each tensor's delta from θ as a CWQUAN1 delta on a uniform step: an f16 scale per 128 entries (the group's RMS
    over c, c the root of the largest, so neither the scales nor the codebook underflow f16) times the nearest of the
    2**bits levels j·step·c, j from -2**(bits-1), zero among them. Returns the entries and the relative error."""
    levels = torch.arange(-2 ** (bits - 1), 2 ** (bits - 1), dtype=torch.float32)
    entries, err, norm = {}, 0.0, 0.0
    for n, t in cur.items():
        d = (t - theta[n].to(t.device).float()).reshape(-1)
        g = math.gcd(d.numel(), 128)
        rms = d.view(-1, g).square().mean(1).sqrt()
        top = float(rms.max())
        c = math.sqrt(top) if top > 0 else 1.0
        sc = (rms / c).half()
        cb = (levels.to(d.device) * (step * c)).half()
        cbf = cb.float()
        z = d.view(-1, g) / torch.where(sc.float() > 0, sc.float(), 1.0)[:, None]
        codes = torch.bucketize(z, (cbf[1:] + cbf[:-1]) / 2)
        q = cbf[codes] * sc.float()[:, None]
        err += float((q.reshape(-1) - d).square().sum())
        norm += float(d.square().sum())
        entries[n] = {"kind": "delta", "q": prepare.QArray(cb, sc, codes.to(torch.uint8), bits, g, tuple(t.shape))}
    return entries, math.sqrt(err / norm) if norm else 0.0


UNIFORM_TRIES, UNIFORM_FILL = 8, 0.98
# a fitted step never goes below UNIFORM_COVER / 2**(bits-1): finer, the outer levels clip the delta's tails
UNIFORM_COVER = 6.0


@torch.no_grad()
def encode_update(store, theta, cur, factors, mats, gains, params, steps, info, max_bytes=None) -> bytes:
    if store.startswith("q"):
        bits, _, grp = store[1:].partition(":")
        bits, grp = int(bits), int(grp or 128)
        mid, half = 2 ** (bits - 1), max(1, 2 ** (bits - 1) - 1)
        entries, err, norm = {}, 0.0, 0.0
        for n, t in cur.items():
            d = (t - theta[n].to(t.device).float()).reshape(-1)
            g = math.gcd(d.numel(), grp)
            top = float(d.abs().max())
            # the range is split between codebook and scales so neither underflows f16
            c = math.sqrt(top) if top > 0 else 1.0
            sc = (d.view(-1, g).abs().amax(1) / c).half()
            cb = ((torch.arange(2 ** bits, device=d.device) - mid) / half * c).half()
            step = sc.float() * (c / half)
            codes = (d.view(-1, g) / torch.where(step > 0, step, 1.0)[:, None]).round().add(mid).clamp(0, 2 ** bits - 1)
            q = cb.float()[codes.long()] * sc.float()[:, None]
            err += float((q.reshape(-1) - d).square().sum())
            norm += float(d.square().sum())
            entries[n] = {"kind": "delta", "q": prepare.QArray(cb, sc, codes.to(torch.uint8), bits, g, tuple(t.shape))}
        info.update(bits=bits, group=grp, rel_err=math.sqrt(err / norm) if norm else 0.0)
        steps.append({"op": "extra", "flops": 8 * sum(t.numel() for t in cur.values()), "what": "qB codes of the deltas"})
        return prepare.encode_quant(entries, theta)
    if store.startswith("u"):
        bits, _, s = store[1:].partition(":")
        bits, s = int(bits), float(s or 1.0)
        numel = sum(t.numel() for t in cur.values())
        tries = 0
        if max_bytes is not None:
            # the step is in RMS units, so the last checkpoint's fitted step starts the search
            floor = UNIFORM_COVER / 2 ** (bits - 1)
            s, best = max(floor, info.get("step", s)), None
            for tries in range(1, UNIFORM_TRIES + 1):
                size = quant_size(uniform_code(theta, cur, bits, s)[0], theta)
                if size <= max_bytes and (best is None or size > best[0]):
                    best = (size, s)
                if UNIFORM_FILL * max_bytes <= size <= max_bytes or (size <= max_bytes and s <= floor):
                    break
                # each doubling of the step takes about a bit a value off an entropy-coded uniform code
                s = max(floor, s * 2 ** (8 * (size - (1 + UNIFORM_FILL) / 2 * max_bytes) / numel))
            if best is None:
                raise SystemExit(f"shallow: {store} fits no step in --max-e-bytes {max_bytes} ({size} B at {s:.3g})")
            s = best[1]
        entries, rel = uniform_code(theta, cur, bits, s)
        blob = prepare.encode_quant(entries, theta)
        if max_bytes is not None and len(blob) > max_bytes:
            raise SystemExit(f"shallow: quant_size no longer matches prepare.encode_quant ({len(blob)} B > {max_bytes})")
        info.update(bits=bits, group=128, codebook="uniform", step=s, tries=tries, rel_err=rel)
        steps.append({"op": "extra", "flops": 8 * numel * (tries + 1), "what": "uB codes of the deltas (uniform RMS step)"})
        return blob
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
