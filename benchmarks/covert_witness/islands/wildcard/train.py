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
- `cached`: a set in the top blocks trained over θ's cached activations with a top-K head (`train_cached`).
- `scale`: every tensor of the set times one line-searched scalar.
- `--extrapolate`: after training, θ + αΔ with α line-searched (ExPO); `--suite`: several configs in one run.

Codes (witnesses for formalism v1's cut cost a, `adamw` / `muon`):
- `--cut B:q4,...`: the residual stream at block boundary B is handed on as a code (`Coded`): a randomized DCT, rebuilt
  in every step, then each group's coordinates on a 2^q-level grid with a bf16 scale; the gradient crosses back in the
  same code with stochastic rounding (`q2b4`: 2 bits forward, 4 back; `g64`: a scale per 64 coordinates; `q4o4`: each
  token's 4 largest coordinates go apart at bf16 with their indices, and the rest is coded with its own scale). Training
  sees only the coded state; the evaluator scores the uncoded model, and the probe windows are scored both ways.
- `--cut B:dq2,...`: the forward codes only the update's effect h - h_frozen at B; h_frozen, θ's residual for the same
  tokens, comes from frozen copies of the trained blocks (the receiving side recomputes it from the token's data), and
  its products are counted and declared as the code's.
- `--lora-rank r --lora-fix B --z-bits q`: a rank-r update whose B is r seeded orthonormal columns, so its whole effect
  on a token is r coefficients z = A·h, handed on (and their gradient back) at q bits each. `--lora-fix A` trains only
  B over a frozen random A.

`--class` must be the class launch.py runs it under (the submission names it); its source rides in the submission's
notes. Every method marks explicit harness steps (`hooks.step`), so it runs in the harness's script mode:

    python benchmarks/covert_witness/launch.py shallow s50 --note "cachedhead ..." -- --class shallow --set mlp-in:-1 --method cached
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import re
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import prepare

try:
    from te_core import hooks
except ImportError:
    hooks = None

KLASS = "reference"
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


class Skip(Exception):
    """A config that cannot run within its budget, raised before it computes anything: a suite goes on to the next."""


def probe_flops(a, rec, trainable, n_values: int) -> int:
    tokens = a.batch * prepare.EVAL_CTX
    return PROBE_BATCHES * n_values * prepare.train_flops(rec, trainable, [{"op": "forward", "tokens": tokens}])["flops"]


def line_search(a, fwd, data, steps: list, apply, values: list[float]) -> tuple[float, dict]:
    """Mean loss on PROBE_BATCHES fresh training batches (each logged as a forward pass) at every value `apply` sets; the
    least wins and stays applied."""
    tokens = a.batch * prepare.EVAL_CTX
    results = {}
    with step_of():
        probe = [data.next(a.batch) for _ in range(PROBE_BATCHES)]
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


FRESH: dict = {}
CUT = {"on": True}


def rotation(d: int, seed: int, device) -> torch.Tensor:
    """An orthogonal d x d matrix, the orthonormal DCT-II times seeded random signs, made from factory ops."""
    i = torch.arange(d, device=device, dtype=torch.float32)
    m = torch.cos(math.pi * (i[None, :] + 0.5) * i[:, None] / d) * math.sqrt(2.0 / d)
    m[0] = m[0] / math.sqrt(2.0)
    g = torch.Generator(device=device).manual_seed(seed)
    return m * (torch.randint(0, 2, (d,), generator=g, device=device).float() * 2 - 1)


def fresh_rotation(d: int, seed: int, device) -> torch.Tensor:
    key = ("rot", d, seed, str(device))
    if key not in FRESH:
        FRESH[key] = rotation(d, seed, device)
    return FRESH[key]


#: The optimal uniform quantizer's step for a unit Gaussian at 1-3 bits (Max 1960); rotated coordinates are near-Gaussian.
GAUSS_STEP = {1: 1.596, 2: 0.9957, 3: 0.5860}


def quantize(y: torch.Tensor, bits: int, group: int, stochastic: bool) -> torch.Tensor:
    """Each group of `group` trailing coordinates on a grid of 2^bits levels with one bf16 scale per group. Nearest
    rounding (forward) uses the mid-rise grid (k + 1/2)·step, step GAUSS_STEP times the group's RMS at 1-3 bits and
    2·absmax / 2^bits above. Stochastic rounding (gradients) uses the grid whose end levels are ±absmax (rounded up), so
    no value is clipped and the code is unbiased."""
    shape = y.shape
    y = y.reshape(-1, shape[-1] // group, group)
    if stochastic:
        top = (y.abs().amax(-1, keepdim=True) * (1 + 2 ** -7)).to(torch.bfloat16).float().clamp_min(1e-30)
        step = 2 * top / (2 ** bits - 1)
        j = torch.floor((y + top) / step + torch.rand_like(y)).clamp(0, 2 ** bits - 1)
        return (j * step - top).reshape(shape)
    if bits in GAUSS_STEP:
        step = y.pow(2).mean(-1, keepdim=True).sqrt() * GAUSS_STEP[bits]
    else:
        step = y.abs().amax(-1, keepdim=True) * (2.0 / 2 ** bits)
    step = step.to(torch.bfloat16).float().clamp_min(1e-30)
    half = 2 ** (bits - 1)
    return ((torch.floor(y / step).clamp(-half, half - 1) + 0.5) * step).reshape(shape)


def code_bits(d: int, bits: int, group: int) -> int:
    """Bits per token of one coded d-vector: `bits` per coordinate and a bf16 step per group."""
    return d * bits + 16 * (d // group)


class Coded(torch.autograd.Function):
    """A token's d-vector handed across a cut as a code: rotated by R, quantized per group, rotated back (forward,
    nearest rounding); the backward passes the gradient straight through the quantizer and hands it across the cut in
    the same code with stochastic rounding. With `rotate` false the vector is coded in its own basis. With `outliers`
    k > 0 the forward first takes out each token's k largest coordinates, which go apart at bf16 (with their indices),
    and codes the rest."""

    @staticmethod
    def forward(ctx, x, bits_f, bits_b, group, seed, rotate, outliers=0):
        ctx.cfg = (bits_b, group, seed, rotate)
        with torch.autocast(x.device.type, enabled=False):
            y = x.float()
            if outliers:
                idx = y.abs().topk(outliers, dim=-1).indices
                val = y.gather(-1, idx).to(torch.bfloat16).float()
                y = y.scatter(-1, idx, 0.0)
            if rotate:
                r = fresh_rotation(x.shape[-1], seed, x.device)
                y = quantize(y @ r.T, bits_f, group, False) @ r
            else:
                y = quantize(y, bits_f, group, False)
            if outliers:
                y = y.scatter(-1, idx, val)
            return y.to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        bits_b, group, seed, rotate = ctx.cfg
        with torch.autocast(g.device.type, enabled=False):
            y = g.float()
            if rotate:
                r = fresh_rotation(g.shape[-1], seed, g.device)
                out = quantize(y @ r.T, bits_b, group, True) @ r
            else:
                out = quantize(y, bits_b, group, True)
        return out.to(g.dtype), None, None, None, None, None, None


CUT_RE = re.compile(r"(d?)q(\d+)(?:o(\d+))?(?:b(\d+))?(?:g(\d+))?")


def parse_cut(spec: str, L: int, d: int, low: int | None = None) -> list[dict]:
    """`B:CODE,...`: boundary B (1..L; the input of block B, L the input of the final norm; `all` every one) handed on
    as CODE = [d]q<forward bits>[o<k>][b<backward bits>][g<group>] in a randomized-DCT basis (group d unless given);
    `o<k>` sends each token's k largest coordinates apart, a bf16 value and a ceil(log2 d)-bit index each. With the `d`
    prefix the forward codes only the update's effect h - h_frozen, where h_frozen is θ's residual at B for the same
    tokens, which the receiving side recomputes from the token's data; it needs B above the lowest trained block `low`
    (at or below it the effect is 0)."""
    out = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        where, code = part.split(":")
        m = CUT_RE.fullmatch(code)
        if not m:
            raise SystemExit(f"wildcard: --cut code {code!r} is not [d]q<bits>[o<k>][b<bits>][g<group>]")
        delta = m.group(1) == "d"
        bf = int(m.group(2))
        k = int(m.group(3) or 0)
        bb = int(m.group(4) or bf)
        group = int(m.group(5) or d)
        if d % group or not (1 <= bf <= 16 and 1 <= bb <= 16) or not 0 <= k < d:
            raise SystemExit(f"wildcard: --cut {part!r}: bits 1-16, outliers below d and a group dividing d = {d}")
        side = k * (16 + math.ceil(math.log2(d)))
        for b in (range(1, L + 1) if where == "all" else [int(where)]):
            if not 1 <= b <= L:
                raise SystemExit(f"wildcard: --cut boundary {b} is not in 1..{L}")
            if delta and (low is None or not 0 <= low < b):
                raise SystemExit(f"wildcard: --cut {b}:{code} codes h - h_frozen, which needs trained blocks only, "
                                 f"the lowest below boundary {b} (lowest trained block: {low})")
            out.append({"boundary": b, "bits_fwd": bf, "bits_bwd": bb, "group": group, "delta": delta, "outliers": k,
                        "a_fwd_bits": code_bits(d, bf, group) + side, "a_bwd_bits": code_bits(d, bb, group)})
    return sorted(out, key=lambda c: c["boundary"])


def frozen_blocks(net, theta: dict, lo: int, hi: int) -> dict:
    """Copies of blocks lo..hi-1 holding θ's tensors exactly, without hooks or gradients (the counter takes an exact
    copy of a θ tensor for θ's own weight, so their products are frozen work)."""
    out = {}
    for j in range(lo, hi):
        blk = copy.deepcopy(net.inner.blocks[j])
        for m in blk.modules():
            m._forward_hooks.clear()
            m._forward_pre_hooks.clear()
        for pname, p in blk.named_parameters():
            p.requires_grad_(False)
            p.data = theta[f"blocks.{j}.{pname}"].to(p.device).clone()
        out[j] = blk
    return out


def frozen_flops(rec: dict, lo: int, hi: int) -> int:
    """Forward FLOPs per token of blocks lo..hi-1."""
    names = tuple(f"blocks.{j}." for j in range(lo, hi))
    mm = sum(n["flops"] for n in prepare.positions(rec) if n["param"].startswith(names))
    return mm + (hi - lo) * prepare.attention_flops(rec, prepare.EVAL_CTX)


def install_cuts(net, cuts: list[dict], theta: dict | None = None, low: int | None = None):
    """Each boundary's code as a forward hook on the block below it (boundary B: block B - 1's output), active while
    CUT["on"]. A delta code's hook adds h_frozen back after coding h - h_frozen; h_frozen is carried up from block
    `low`'s input through frozen copies of blocks low.. (frozen_blocks)."""
    by_block = {c["boundary"] - 1: c for c in cuts}
    top = max((c["boundary"] for c in cuts if c["delta"]), default=None)
    frozen = frozen_blocks(net, theta, low, top) if top is not None else {}
    state = {}
    handles = []
    for j in sorted(set(by_block) | set(frozen)):
        def hook(m, inp, out, j=j, c=by_block.get(j)):
            if not CUT["on"]:
                return out
            f = None
            if j in frozen:
                with torch.no_grad():
                    f = state[j + 1] = frozen[j](state.pop(j) if j > low else inp[0].detach(), inp[1], inp[2])
            if c is None:
                return out
            seed = 1000 + c["boundary"]
            if CUT.get("spectrum") is not None:
                with torch.no_grad():
                    acc = CUT["spectrum"].setdefault(c["boundary"], {"of": "h - h_frozen" if c["delta"] else "h"})
                    spectrum_add(acc, out - f if c["delta"] else out, out)
            if c["delta"]:
                return f + Coded.apply(out - f, c["bits_fwd"], c["bits_bwd"], c["group"], seed, True, c["outliers"])
            return Coded.apply(out, c["bits_fwd"], c["bits_bwd"], c["group"], seed, True, c["outliers"])
        handles.append(net.inner.blocks[j].register_forward_hook(hook))
    return handles


SESSION = 4096
RANKS = (4, 8, 16, 32, 64, 128)
COORDS = (1, 2, 4, 8, 16, 32)


def spectrum_add(acc: dict, delta: torch.Tensor, h: torch.Tensor):
    """Accumulate, per SESSION-token session of `delta` (windows, ctx, d; δ = h for a whole-state code): the share of
    its energy that its top r directions hold (eigenvalues λ of δᵀδ), the transform-coding gain ½·log2(mean λ / geomean
    λ) (bits per coordinate a fixed basis with bits allocated by variance saves over a flat code at high rate) and the
    same over the coordinates' own variances (`channel_gain_bits`, the gain without a basis), the share of a session's
    energy in its largest token, the share of a token's energy in its largest k coordinates, and |δ|² against |h|²."""
    d = delta.shape[-1]
    e = delta.float().pow(2)
    tok = e.sum(-1)
    acc["delta_sq"] = acc.get("delta_sq", 0.0) + float(tok.sum())
    acc["h_sq"] = acc.get("h_sq", 0.0) + float(h.float().pow(2).sum())
    if not float(tok.sum()):
        return
    top = e.topk(min(max(COORDS), d), dim=-1).values.cumsum(-1) / tok.unsqueeze(-1).clamp_min(1e-30)
    acc["tokens"] = acc.get("tokens", 0) + tok.numel()
    for k in COORDS:
        if k <= d:
            acc[f"coord_top{k}"] = acc.get(f"coord_top{k}", 0.0) + float(top[..., k - 1].sum())
    per = min(max(1, SESSION // delta.shape[1]), delta.shape[0])
    n = delta.shape[0] // per * per
    x = delta[:n].float().reshape(n // per, -1, d)
    with torch.autocast(x.device.type, enabled=False):
        ev = torch.linalg.eigvalsh(x.transpose(1, 2) @ x).flip(-1).clamp_min(0)
    share = ev.cumsum(-1) / ev.sum(-1, keepdim=True).clamp_min(1e-30)
    lam = (ev / ev.mean(-1, keepdim=True).clamp_min(1e-30)).clamp_min(1e-9)
    var = x.pow(2).mean(1)
    var = (var / var.mean(-1, keepdim=True).clamp_min(1e-30)).clamp_min(1e-9)
    peak = x.pow(2).sum(-1)
    acc["sessions"] = acc.get("sessions", 0) + share.shape[0]
    acc["gain_bits"] = acc.get("gain_bits", 0.0) + float((-0.5 * torch.log2(lam).mean(-1)).sum())
    acc["channel_gain_bits"] = acc.get("channel_gain_bits", 0.0) + float((-0.5 * torch.log2(var).mean(-1)).sum())
    acc["token_peak"] = acc.get("token_peak", 0.0) + float((peak.amax(-1) / peak.sum(-1).clamp_min(1e-30)).sum())
    for r in RANKS:
        if r <= d:
            acc[f"top{r}"] = acc.get(f"top{r}", 0.0) + float(share[:, r - 1].sum())


def spectrum_report(spec: dict) -> dict:
    out = {}
    for b, acc in sorted(spec.items()):
        s, t = acc.get("sessions", 0), acc.get("tokens", 0)
        out[str(b)] = {"of": acc.get("of"), "sessions": s,
                       "delta_rms_over_h_rms": (acc["delta_sq"] / max(acc["h_sq"], 1e-30)) ** 0.5,
                       **{k: v / s for k, v in acc.items()
                          if s and (k.startswith("top") or k in ("gain_bits", "channel_gain_bits", "token_peak"))},
                       **{k: v / t for k, v in acc.items() if t and k.startswith("coord_top")}}
    return out


def lowest_block(tuned: list[str]) -> int | None:
    """The lowest block index a trained tensor sits in (-1 for the embedding, L for the final norm)."""
    found = []
    for n in tuned:
        m = re.match(r"blocks\.(\d+)\.", n)
        found.append(int(m.group(1)) if m else (-1 if n == "embed" else 10 ** 6))
    return min(found) if found else None


def lora_bytes(theta: dict, mats: list[str], gains: list[str], rank: int) -> int:
    fac = {n: (torch.zeros(rank, theta[n].shape[1]), torch.zeros(theta[n].shape[0], rank), 1.0) for n in mats}
    return len(prepare.encode_lora(fac, "bf16", {n: theta[n] for n in gains}))


def add_adapters(net, theta: dict, params: dict, mats: list[str], gains: list[str], rank: int, fix: str = "",
                 zbits: int = 0) -> tuple[dict, int]:
    """Rank-r adapters W + s·B·A on `mats` (A ~ N(0, 1/d_in), B = 0, s = sqrt(d_in / r)); rank -1 is the largest whose E
    fits the class's cap. `fix` "A" freezes that random A (only B trains). `fix` "B" makes B r seeded orthonormal
    columns (rows of a randomized DCT), frozen, with A = 0 trained and s = sqrt(d_out / r), so the update's whole effect
    on a token is its r coefficients z = A·h; `zbits` hands z on coded at that many bits per coefficient (one bf16 scale
    per token), and its gradient back the same way."""
    if rank < 0:
        cap = prepare.CLASSES[KLASS]["max_e_bytes"]
        if cap is None or fix:
            raise SystemExit(f"wildcard: --lora-rank -1 needs a class with an E cap and no --lora-fix, not {KLASS}")
        rank = 1
        while lora_bytes(theta, mats, gains, rank + 1) <= cap - 512:
            rank += 1
    if zbits and fix != "B":
        raise SystemExit("wildcard: --z-bits codes z = A·h, which only --lora-fix B makes the update's whole effect")
    for n in mats:
        params[n].requires_grad_(False)
        params[n].data = theta[n].to(params[n].device).clone()
    g = torch.Generator(device=prepare.DEVICE).manual_seed(0)
    factors = {}
    for i, n in enumerate(mats):
        d_out, d_in = params[n].shape
        if fix == "B":
            pick = torch.randperm(d_out, generator=g, device=prepare.DEVICE)[:rank]
            B = rotation(d_out, 2000 + i, prepare.DEVICE)[pick].T.contiguous()
            A = torch.nn.Parameter(torch.zeros(rank, d_in, device=prepare.DEVICE))
            s = math.sqrt(d_out / rank)
        else:
            A = torch.randn(rank, d_in, device=prepare.DEVICE, generator=g) / math.sqrt(d_in)
            A = A if fix == "A" else torch.nn.Parameter(A)
            B = torch.nn.Parameter(torch.zeros(d_out, rank, device=prepare.DEVICE))
            s = math.sqrt(d_in / rank)
        factors[n] = (A, B, s)

        def hook(m, inp, out, A=A, B=B, s=s):
            z = F.linear(inp[0], A)
            if zbits:
                z = Coded.apply(z, zbits, zbits, rank, 0, False, 0)
            return out + s * F.linear(z, B)
        net.inner.get_submodule(n.removesuffix(".weight")).register_forward_hook(hook)
    return factors, rank


def train_iterative(a, net, rec, theta, params, tuned, data, fwd=None):
    fwd = fwd or net
    mats = [n for n in tuned if params[n].ndim == 2]
    gains = [n for n in tuned if params[n].ndim == 1]
    opts, extra, factors = [], 0, {}
    if a.lora_rank:
        factors, a.lora_rank = add_adapters(net, theta, params, mats, gains, a.lora_rank, a.lora_fix, a.z_bits)
        trainable = {"full": gains, "adapters": {n: a.lora_rank for n in mats}}
        groups = [{"params": [p for A, B, _ in factors.values() for p in (A, B) if isinstance(p, torch.nn.Parameter)],
                   "lr": a.lr},
                  {"params": [params[n] for n in gains], "lr": a.norm_lr}]
        opts.append((torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, a.beta2), weight_decay=a.wd,
                                       fused=prepare.DEVICE.startswith("cuda")), None))
    else:
        trainable = {"full": tuned}
        emb = ["embed"] if a.embed_lr and "embed" in mats else []
        if a.method == "muon" and mats:
            muon = [n for n in mats if n not in emb]
            opts.append((Muon([params[n] for n in muon], lr=a.lr, wd=a.wd), a.lr))
            extra = sum(ns_flops(tuple(params[n].shape)) for n in muon)
            adam_names = emb + gains
        else:
            adam_names = mats + gains
        if adam_names:
            groups = [{"params": [params[n] for n in adam_names if params[n].ndim == 2 and n not in emb], "lr": a.lr},
                      {"params": [params[n] for n in emb], "lr": a.embed_lr},
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
    d = rec["config"]["d_model"]
    low = lowest_block(tuned)
    cuts = parse_cut(a.cut, rec["config"]["n_layer"], d, low) if a.cut else []
    crossed = [c for c in cuts if low is not None and low < c["boundary"]]
    top_delta = max((c["boundary"] for c in cuts if c["delta"]), default=None)
    refresh = frozen_flops(rec, low, top_delta) if top_delta is not None else 0
    code_fwd = 4 * d * d * len(cuts) + refresh
    code_train = code_fwd + 4 * d * d * len(crossed)
    if cuts:
        install_cuts(net, cuts, theta, low)
        print(json.dumps({"cuts": cuts, "gradient_crosses": [c["boundary"] for c in crossed],
                          "code_flops_per_token": {"forward": code_fwd, "train": code_train,
                                                   "frozen_recompute": refresh}}), flush=True)
    cut_probe = probe_rows(rec, a)[: a.cut_probe_rows] if cuts and a.cut_probe_rows else None
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"] + extra
    per_step += code_train * tokens
    alphas = [1.0] + a.extrapolate if a.extrapolate else []
    reserve = probe_flops(a, rec, trainable, len(alphas)) if alphas else 0
    if cut_probe is not None:
        probe_tokens = len(cut_probe) * prepare.EVAL_CTX
        reserve += 4 * prepare.train_flops(rec, trainable, [{"op": "forward", "tokens": probe_tokens}])[
            "flops"] + 2 * code_fwd * probe_tokens
        reserve += 4 * len(cuts) * (probe_tokens * d * d + (probe_tokens // SESSION + 1) * d ** 3)
    n_steps = int((a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - reserve) // per_step)
    if n_steps < 1:
        raise Skip(f"wildcard {a.method}: the budget leaves no step after the line search's reserve")
    base = [[g["lr"] for g in o.param_groups] for o, _ in opts]
    if a.ema and factors:
        raise SystemExit("wildcard: --ema is for full-tensor sets")
    ema = {n: params[n].detach().clone() for n in tuned} if a.ema else None
    steps, snaps = [], []
    marks = {round(n_steps * k / (a.snaps + 1)) for k in range(1, a.snaps + 1)}
    print(json.dumps({"steps": n_steps, "tuned": tuned, "flops_per_step": per_step, "extra": extra}), flush=True)
    probes = {}

    def cut_probes(when: str):
        """The probe windows' loss with every cut coded and with none (the evaluator scores the uncoded model)."""
        with step_of():
            for on in (True, False):
                CUT["on"] = on
                if on and (when == "end" or any(not c["delta"] for c in cuts)):
                    CUT["spectrum"] = {}
                probes[f"{when}_{'coded' if on else 'uncoded'}"] = probe_loss(fwd, cut_probe, a.micro, steps)
                if CUT.get("spectrum") is not None:
                    probes[f"{when}_spectrum"] = spectrum_report(CUT.pop("spectrum"))
                if on:
                    steps.append({"op": "extra", "flops": code_fwd * len(cut_probe) * prepare.EVAL_CTX,
                                  "what": "cut code (probe)"})
            CUT["on"] = True
        print(json.dumps({"cut_probe": when, **probes}), flush=True)

    if cut_probe is not None:
        cut_probes("theta")
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
        with step_of():
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
            if cuts:
                steps.append({"op": "extra", "flops": code_train * tokens, "what": "cut code"})
            if ema is not None:
                with torch.no_grad():
                    for n in tuned:
                        ema[n].lerp_(params[n].detach(), 1 - a.ema)
            if step + 1 in marks:
                snaps.append((encode(ema if ema is not None else params), len(steps)))
                if getattr(a, "probe", None) is not None:
                    print(json.dumps({"step": step + 1, "flops": prepare.train_flops(rec, trainable, steps)["flops"],
                                      "probe_loss": probe_loss(fwd, a.probe, a.micro, steps)}), flush=True)
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
    info = {"steps": n_steps, "lora_rank": a.lora_rank}
    if cuts:
        if cut_probe is not None:
            cut_probes("end")
        info["cuts"] = {"codes": cuts, "gradient_crosses": [c["boundary"] for c in crossed], "probe": probes,
                        "probe_rows": 0 if cut_probe is None else len(cut_probe)}
    if a.z_bits:
        info["z_code"] = {"bits": a.z_bits, "rank": a.lora_rank, "a_bits_per_direction": a.lora_rank * a.z_bits + 16}
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


def train_kfac(a, net, rec, theta, params, tuned, data):
    """`--kfac-rounds` damped K-FAC steps on each trained matrix, each from statistics gathered at the current weights
    over an equal share of the budget and line-searched; the gains take the plain gradient. Each round but the last ends
    in a snapshot."""
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

    trainable = {"full": tuned}
    tokens = a.batch * prepare.EVAL_CTX
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"]
    moment_flops = sum(2 * tokens * (params[n].shape[0] ** 2 + params[n].shape[1] ** 2) for n in mats)
    rounds = a.kfac_rounds
    reserve = rounds * probe_flops(a, rec, trainable, len(a.etas) + 1)
    n_steps = int((a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - reserve) // (per_step + moment_flops)) // rounds
    if n_steps < 1:
        raise Skip("wildcard kfac: the budget leaves no accumulation step after the line searches' reserve")
    steps, snaps, searches = [], [], []
    print(json.dumps({"steps": n_steps, "rounds": rounds, "tuned": tuned, "flops_per_step": per_step,
                      "moment_flops": moment_flops}), flush=True)
    for rnd in range(rounds):
        for s in stats.values():
            s["A"].zero_()
            s["S"].zero_()
        handles = [mods[n].register_forward_hook(lambda m, i, o, n=n: fwd_hook(m, i, o, n)) for n in mats]
        for step in range(n_steps):
            with step_of():
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
                print(json.dumps({"round": rnd, "step": step + 1, "loss": float(loss.detach()) * a.batch / len(x)}),
                      flush=True)
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
        searches.append({"eta": best, "line_search": results})
        if rnd < rounds - 1:
            with step_of():
                snaps.append((prepare.encode_full(delta_of(params, theta, tuned)), len(steps)))
    return (prepare.encode_full(delta_of(params, theta, tuned)), trainable, steps, snaps,
            {"steps": n_steps * rounds, "rounds": searches})


def train_scale(a, net, rec, theta, params, tuned, data):
    """Every tensor of the set times one scalar α, line-searched on fresh batches (the final norm's gain: a temperature);
    a second search refines around the first's best."""
    trainable = {"full": tuned}
    start = {n: params[n].detach().clone() for n in tuned}

    @torch.no_grad()
    def apply(al):
        for n in tuned:
            params[n].copy_(start[n] * al)

    steps = []
    best, coarse = line_search(a, net, data, steps, apply, [1.0] + a.scales)
    width = min([abs(x - best) for x in [1.0] + a.scales if x != best] or [0.05]) / 2
    best, fine = line_search(a, net, data, steps, apply, [best, best - width, best + width, best - width / 2,
                                                          best + width / 2])
    return (prepare.encode_full(delta_of(params, theta, tuned)), trainable, steps, [],
            {"alpha": best, "coarse": coarse, "fine": fine})


CACHED_OK = ("w_gate", "w_up", "w_down")


def train_cached(a, net, rec, theta, params, tuned, data):
    """A set in the top blocks trained over cached activations with a top-K head.

    θ's prefix below the set is frozen, so the residual stream entering it is the same at every step: each fresh
    position is run once through θ (a logged forward), and that activation is cached with θ's top-K tokens there, the
    rest's probability-weighted mean embedding ē and c = lse_rest(θ) − ē·z(θ). Training steps then run only the cached
    part and K + 2 head rows: loss = lse(top-K logits, target, c + ē·z) − target logit, which has θ's exact gradient at θ
    (the rest's log-mass to first order in z). A set within the top block's MLP (and the final gain) caches x_mid, the
    residual entering that MLP, and samples single positions; any other set caches the residual entering its lowest
    block for whole windows and runs blocks from there (attention needs the window). Each chunk of fresh positions is
    followed by its share of the steps, and its end is a snapshot. Every cached-step FLOP is logged as `extra`, counted
    as the meter counts a train pass above the lowest trained node (every matmul's input gradient, the trained
    matrices' weight gradients), with attention at its full square."""
    inner = net.inner
    L, d, V, T = len(inner.blocks), inner.size.d_model, inner.vocab, prepare.EVAL_CTX
    top = inner.blocks[L - 1]
    ok = {f"blocks.{L - 1}.{leaf}.weight" for leaf in CACHED_OK} | {"norm"}
    if "embed" in tuned:
        raise SystemExit("wildcard cached: the tied embedding is read below every cached activation")
    seq = not set(tuned) <= ok
    start = min((int(n.split(".")[1]) for n in tuned if n.startswith("blocks.")), default=L - 1)
    if seq and a.sample != "uniform":
        raise Skip("wildcard cached: --sample loss draws positions; a window cache draws whole windows")
    mats = [n for n in tuned if params[n].ndim == 2]
    gains = [n for n in tuned if params[n].ndim == 1]
    K, P = a.top_k, a.batch * T
    d_ff = top.w_gate.weight.shape[0]
    head = 2 * (K + 2) * d
    if seq:
        attn = 2 * prepare.attention_flops(rec, T)
        fwd_mm = (L - start) * (2 * (4 * d * d + 3 * d * d_ff) + attn) + head
        per_pos = 2 * fwd_mm + (L - start) * attn + sum(2 * params[n].numel() for n in mats)
    else:
        fwd_mm = 2 * 3 * d * d_ff + head
        per_pos = 2 * fwd_mm + sum(2 * params[n].numel() for n in mats)
    mic = max(T, a.cached_micro // T * T) if seq else a.cached_micro
    trainable = {"full": tuned}
    fill_tok = prepare.train_flops(rec, trainable, [{"op": "forward", "tokens": 1}])["flops"] + 2 * V * d
    n_pos = int(a.budget_fraction * prepare.TRAIN_FLOP_BUDGET // (fill_tok + a.epochs * per_pos))
    chunks = a.snaps + 1
    seqs = max(chunks, n_pos // prepare.EVAL_CTX) // chunks * chunks
    n_pos = seqs * prepare.EVAL_CTX
    n_steps = max(chunks, int(a.epochs * n_pos) // P) // chunks * chunks
    gb = n_pos * (4 * d + 2 * K + 13) / 2 ** 30
    print(json.dumps({"positions": n_pos, "steps": n_steps, "cache_gb": round(gb, 2), "per_position_step": per_pos,
                      "fill_per_token": fill_tok, "top_k": K}), flush=True)
    if gb > a.cache_gb:
        raise Skip(f"wildcard cached: the cache needs {gb:.1f} GB > --cache-gb {a.cache_gb}")
    dev = prepare.DEVICE
    xs = torch.empty(n_pos, d, dtype=torch.bfloat16, device=dev)
    es = torch.empty(n_pos, d, dtype=torch.bfloat16, device=dev)
    ids = torch.empty(n_pos, K, dtype=torch.int16, device=dev)
    tgt = torch.empty(n_pos, dtype=torch.int32, device=dev)
    cs = torch.empty(n_pos, dtype=torch.float32, device=dev)
    dup = torch.empty(n_pos, dtype=torch.bool, device=dev)
    ce0 = torch.empty(n_pos, dtype=torch.float32, device=dev)
    grab = {}
    handles = [inner.blocks[start].register_forward_pre_hook(lambda m, i: grab.__setitem__("x", i[0])),
               top.register_forward_hook(lambda m, i, o: grab.__setitem__("y", o))]
    if not seq:
        handles.append(top.wo.register_forward_hook(lambda m, i, o: grab.__setitem__("o", o)))
    embed = inner.embed.detach()
    cos, sin = inner.cos[:T], inner.sin[:T]

    @torch.no_grad()
    def fill(lo: int, hi: int, steps: list):
        at = lo
        while at < hi:
            with step_of():
                at = fill_micro(at, hi, steps)

    def fill_micro(at: int, hi: int, steps: list) -> int:
        if True:
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
            ebar = ((p @ embed) / mass[:, None]).bfloat16()
            del p
            # the cache is persistent and not θ's, so the counter takes any row of it read without a token index
            # for an updated weight: a fill only writes it
            xs[at: at + n] = (grab["x"] if seq else grab["x"] + grab["o"]).reshape(n, d).bfloat16()
            es[at: at + n] = ebar
            ids[at: at + n] = (k_ids - 32768).to(torch.int16)
            tgt[at: at + n] = t.int()
            cs[at: at + n] = lse + mass.log() - (y * ebar.float()).sum(-1)
            ce0[at: at + n] = lse - logits_t
            dup[at: at + n] = (k_ids == t[:, None]).any(-1)
            steps += [{"op": "forward", "tokens": n}, {"op": "extra", "flops": 2 * V * d * n, "what": "cache: rest mean"}]
            return at + n

    def cached_loss(ix: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
        x = xs[ix].float()
        with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
            if seq:
                x = x.view(-1, T, d)
                for b in inner.blocks[start:]:
                    x = b(x, cos, sin)
                y = x.reshape(-1, d)
            else:
                h = F.rms_norm(x, (d,), top.mlp_norm)
                y = x + top.w_down(F.silu(top.w_gate(h)) * top.w_up(h))
        z = rms(y.float(), inner.norm)
        sub = torch.einsum("pd,pkd->pk", z, embed[ids[ix].int() + 32768])
        own = (z * embed[tgt[ix]]).sum(-1)
        rest = cs[ix] + (z * es[ix].float()).sum(-1)
        own_slot = torch.where(dup[ix], torch.full_like(own, -torch.inf), own)
        lse = torch.logsumexp(torch.cat([sub, own_slot[:, None], rest[:, None]], dim=-1), dim=-1)
        return (lse - own).mean() if w is None else ((lse - own) * w).mean()

    groups = [{"params": [params[n] for n in mats], "lr": a.lr}, {"params": [params[n] for n in gains], "lr": a.norm_lr}]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, a.beta2), weight_decay=a.wd, fused=dev.startswith("cuda"))
    base = [g["lr"] for g in opt.param_groups]
    trained = [p for g in opt.param_groups for p in g["params"]]
    steps, snaps = [], []
    gen = torch.Generator(device=dev).manual_seed(a.data_seed)
    per_chunk, step_per_chunk, filled, step = n_pos // chunks, n_steps // chunks, 0, 0
    t0 = time.perf_counter()
    if seq and per_chunk < 2 * T:
        raise Skip("wildcard cached: a window cache needs two windows a chunk (one held out)")
    m = max(T, min(per_chunk // 2, mic) // T * T) if seq else min(per_chunk // 2, mic)
    held = torch.arange(m, device=dev)
    window = torch.arange(T, device=dev)

    @torch.no_grad()
    def held_loss(what: str) -> float:
        steps.append({"op": "extra", "flops": fwd_mm * m, "what": f"cached: held-out loss, {what}"})
        with step_of():
            return float(cached_loss(tokens_of(held)))

    def cached_step(step: int, filled: int, imp) -> float:
        frac = min(1.0, (step + 1) / a.warmup) * (1 - step / n_steps)
        for g, lr in zip(opt.param_groups, base):
            g["lr"] = lr * frac
        if seq:
            sq = m // T + torch.randint((filled - m) // T, (P // T,), device=dev, generator=gen)
            ix, w = (sq[:, None] * T + window).reshape(-1), None
        elif imp is not None:
            cdf, floor, mean = imp
            j = torch.searchsorted(cdf, torch.rand(P, device=dev, generator=gen) * cdf[-1]).clamp_max(len(cdf) - 1)
            ix, w = j + m, None
        else:
            ix, w = m + torch.randint(filled - m, (P,), device=dev, generator=gen), None
        ix = tokens_of(ix)
        if imp is not None:
            w = mean / ce0[ix].clamp_min(floor)
        tot = 0.0
        for i in range(0, P, mic):
            wi = None if w is None else w[i: i + mic]
            loss = cached_loss(ix[i: i + mic], wi) * (len(ix[i: i + mic]) / P)
            loss.backward()
            tot += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trained, a.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps.append({"op": "extra", "flops": per_pos * P,
                      "what": (f"cached step: blocks {start}-{L - 1}" if seq else "cached step: top MLP") + ", top-K head"})
        return tot

    for c in range(chunks):
        fill(filled, filled + per_chunk, steps)
        filled += per_chunk
        if c == 0:
            print(json.dumps({"cached_loss_at_theta": held_loss("at theta"), "true_loss": float(ce0[:m].mean()),
                              "held_out_positions": m}), flush=True)
        imp = None
        if a.sample == "loss":
            floor = 0.05 * float(ce0[m:filled].mean())
            q = ce0[m:filled].clamp_min(floor)
            imp = (q.cumsum(0), floor, float(q.mean()))
        for _ in range(step_per_chunk):
            with step_of():
                loss = cached_step(step, filled, imp)
            step += 1
            if step % 100 == 0:
                print(json.dumps({"step": step, "chunk": c, "loss": loss, "s": round(time.perf_counter() - t0, 1)}),
                      flush=True)
        row = {"chunk": c, "filled": filled, "held_out_loss": held_loss(f"chunk {c}")}
        if getattr(a, "probe", None) is not None:
            with step_of():
                row |= {"flops": prepare.train_flops(rec, trainable, steps)["flops"],
                        "probe_loss": probe_loss(net, a.probe, a.micro, steps)}
        print(json.dumps(row), flush=True)
        if c < chunks - 1:
            snaps.append((prepare.encode_full(delta_of(params, theta, tuned)), len(steps)))
    for h in handles:
        h.remove()
    return (prepare.encode_full(delta_of(params, theta, tuned)), trainable, steps, snaps,
            {"steps": n_steps, "positions": n_pos, "epochs": n_steps * P / n_pos, "top_k": K})


@contextlib.contextmanager
def step_of():
    """An explicit harness step (a cached step reads no fresh batch); outside the harness, nothing. Values cached in
    FRESH live for one step, so the counter sees them made in the step that uses them."""
    FRESH.clear()
    with hooks.step() if hooks is not None else contextlib.nullcontext():
        yield
    FRESH.clear()


def tokens_of(ix: torch.Tensor) -> torch.Tensor:
    """Cached positions are token data: their rows of the cache are this step's token-indexed values."""
    return hooks.tokens(ix) if hooks is not None else ix


def rms(x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), gain.float())


def main() -> int:
    global KLASS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mechanism", default="", help="the --note's mechanism name, kept in the notes")
    ap.add_argument("--set", dest="tset", default="mlp-in:-1")
    ap.add_argument("--allow-deep", action="store_true")
    ap.add_argument("--method", default="adamw", choices=("adamw", "muon", "kfac", "cached", "scale"))
    ap.add_argument("--lora-rank", type=int, default=0,
                    help="train the set's matrices through rank-r adapters (E `encode_lora`); -1 fits the class's cap")
    ap.add_argument("--lora-fix", default="", choices=("", "A", "B"),
                    help="freeze the adapters' A (random) or B (seeded orthonormal, A = 0 trained)")
    ap.add_argument("--z-bits", type=int, default=0, help="with --lora-fix B: hand z = A·h on at this many bits per coefficient")
    ap.add_argument("--cut", default="", help="B:[d]q<bits>[o<k>][b<bits>][g<group>],...: block boundaries handed on coded "
                                            "(parse_cut)")
    ap.add_argument("--cut-probe-rows", type=int, default=128,
                    help="with --cut: probe windows scored coded and uncoded at theta and at the end (0: none)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--norm-lr", type=float, default=1e-3)
    ap.add_argument("--embed-lr", type=float, default=0.0,
                    help="the tied embedding's own AdamW rate when it trains (0: it goes with the other matrices)")
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--schedule", default="linear", choices=("linear", "wsd", "const"))
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--ema", type=float, default=0.0, help="snapshot and submit an EMA of the weights (decay per step)")
    ap.add_argument("--damping", type=float, default=1e-2)
    ap.add_argument("--kfac-rounds", type=int, default=1, help="kfac: statistics-then-step rounds over equal budget shares")
    ap.add_argument("--etas", default="1e-4,3e-4,1e-3,3e-3,1e-2", help="kfac: per-element RMS step sizes searched")
    ap.add_argument("--extrapolate", default="", help="adamw/muon: α values searched for θ + α Δ after training (1 is kept)")
    ap.add_argument("--scales", default="0.9,0.95,1.05,1.1,1.2", help="scale: α values searched (1 is kept)")
    ap.add_argument("--top-k", type=int, default=256, help="cached: θ's top-K tokens kept per position")
    ap.add_argument("--epochs", type=float, default=8.0, help="cached: training positions per cached position")
    ap.add_argument("--cache-gb", type=float, default=80.0)
    ap.add_argument("--cached-micro", type=int, default=8192, help="cached: positions per backward")
    ap.add_argument("--sample", default="uniform", choices=("uniform", "loss"),
                    help="cached: draw positions uniformly, or in proportion to θ's loss there with importance weights")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--snaps", type=int, default=7)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    ap.add_argument("--suite", default="", help="a JSON list of argument overrides run one after another in this run; "
                    "each is scored on the same probe windows and the least probe loss is submitted (no snapshots)")
    ap.add_argument("--probe-rows", type=int, default=256, help="windows at the end of the training order kept as a probe")
    ap.add_argument("--class", dest="klass", default=KLASS, choices=tuple(prepare.CLASSES),
                    help="the class launch.py names, which the submission must name too")
    a = ap.parse_args()
    KLASS = a.klass
    torch.backends.cuda.matmul.allow_tf32 = True
    configs = json.loads(a.suite) if a.suite else [{}]
    if a.suite and a.lora_rank:
        raise SystemExit("wildcard: a suite trains full tensors (adapters' hooks would outlive their config)")
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
    probe = probe_rows(rec, a) if a.suite else None
    all_steps, results, best = [], [], None
    for i, over in enumerate(configs):
        c = argparse.Namespace(**(vars(a) | {k.replace("-", "_"): v for k, v in over.items()}))
        if c.tset != a.tset or c.lora_rank != a.lora_rank:
            raise SystemExit("wildcard: a suite keeps one --set and --lora-rank, so its trainable is one update's")
        for key in ("etas", "extrapolate", "scales"):
            v = getattr(c, key)
            setattr(c, key, [float(x) for x in v.split(",") if x] if isinstance(v, str) else list(v))
        c.probe = probe
        with torch.no_grad():
            for n in tuned:
                params[n].copy_(theta[n])
        data = prepare.TrainData(rec["tokenizer"], seed=c.data_seed)
        try:
            blob, trainable, steps, snaps, info = TRAIN.get(c.method, train_iterative)(c, net, rec, theta, params,
                                                                                       tuned, data)
        except Skip as e:
            if not a.suite:
                raise SystemExit(str(e))
            print(json.dumps({"config": i, "overrides": over, "skipped": str(e)}), flush=True)
            results.append({"overrides": over, "skipped": str(e)})
            continue
        if probe is not None:
            if data.at + a.probe_rows > len(data.order):
                info["probe_overlaps_training"] = True
            with step_of():
                info["probe_loss"] = probe_loss(net, probe, c.micro, steps)
            print(json.dumps({"config": i, "overrides": over, "probe_loss": info["probe_loss"], "info": info}), flush=True)
        all_steps += steps
        results.append({"overrides": over, "info": info, "E_bytes": len(blob)})
        if best is None or (probe is not None and info["probe_loss"] < best[-1]):
            best = (c, blob, trainable, snaps, info, info.get("probe_loss"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if best is None:
        raise SystemExit("wildcard: every config of the suite was skipped")
    c, blob, trainable, snaps, info, _ = best
    model = prepare.decode(theta, blob)
    changed = [n for n in tuned if not prepare.same_bits(model[n], theta[n])]
    notes = {"direction": "wildcard", "mechanism": a.mechanism, "method": c.method, "set": tuned,
             "args": {k: v for k, v in vars(c).items() if k != "probe"}, "info": info, "E_bytes": len(blob),
             "suite": results if a.suite else None, "source": Path(__file__).read_text()}
    prepare.submit(Path(a.out), a.start, KLASS, blob, model, trainable, all_steps, changed=changed, notes=notes,
                   snapshots=snaps if not a.suite else [])
    return 0

def probe_rows(rec: dict, a) -> torch.Tensor:
    """The last `--probe-rows` windows of the training order, which a run reaches only after reading all the others."""
    d = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    c = d.ctx
    idx = np.sort(d.order[-a.probe_rows:])
    rows = np.stack([d.tokens[i * c: i * c + c + 1] for i in idx]).astype(np.int64)
    return torch.from_numpy(rows).to(prepare.DEVICE)


@torch.no_grad()
def probe_loss(fwd, rows: torch.Tensor, micro: int, steps: list) -> float:
    tot = 0.0
    for i in range(0, len(rows), micro):
        tot += float(loss_of(fwd, rows[i: i + micro])) * len(rows[i: i + micro])
    steps.append({"op": "forward", "tokens": len(rows) * prepare.EVAL_CTX})
    return tot / len(rows)



TRAIN = {"kfac": train_kfac, "scale": train_scale, "cached": train_cached}


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit as e:
        if e.code is not None and not isinstance(e.code, int):
            print(f"wildcard: exit: {e.code}", flush=True)
        raise
    except Exception:
        traceback.print_exc()
        raise
