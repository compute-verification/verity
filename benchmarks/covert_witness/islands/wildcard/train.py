"""Island `wildcard`: training shaped against formalism v2's Slowdown, through its cut cost a and update profile u.

`--set` names what trains: one updated layer under a frozen tail (u(x) = 0 for the tail's share t of a token's affected
work), the top blocks, or every tensor. `--cut` trains through activation codes at block boundaries, each a witness for
the a its bits per token measure.

Methods:
- `adamw` / `muon`: train the set in full from θ for the budget; E is the trained tensors in full (`encode_full`).
- `kfac`: closed form. Accumulate, over the budget's tokens at θ, each trained matrix's gradient G and its input and
  output-gradient second moments A, S; then one natural-gradient step W − η S⁻¹ G A⁻¹ (damped, scaled to unit RMS per
  element), η picked by a line search on fresh training batches.
- `cached`: a set in the top blocks trained over θ's cached activations with a top-K head (`train_cached`).
- `scale`: every tensor of the set times one line-searched scalar.
- `--extrapolate`: after training, θ + αΔ with α line-searched (ExPO); `--suite`: several configs in one run.

Codes (witnesses for the cut cost a, `adamw` / `muon`):
- `--cut B:q4,...`: the residual stream at block boundary B is handed on as a code (`Coded`): a randomized DCT, rebuilt
  in every step, then each group's coordinates on a 2^q-level grid with a bf16 scale; the gradient crosses back in the
  same code with stochastic rounding (`q2b4`: 2 bits forward, 4 back; `g64`: a scale per 64 coordinates; `q4o4`: each
  token's 4 largest coordinates go apart at bf16 with their indices, and the rest is coded with its own scale; `q4m`: the
  state coded about the boundary's running mean from earlier batches, on the Gaussian grid; `q2b2s`: the gradient
  stochastically rounded on a grid scaled by its RMS, clipped beyond the end levels; `q5meb2e`: either direction
  entropy-coded, a fixed number of bits a token on an unclipped grid whose symbol model learns from earlier batches,
  `ecode`; `q4mr`: the forward Rice-coded into d·bits bits, `rcode`). Training sees only the coded state; the
  evaluator scores the uncoded model, and the probe windows are scored both ways. A fixed forward code's symbols and
  an `r` code's bytes pass through a uint8 tensor at their width, so a traced step's narrowest cut can be the code;
  an `e` code's length is computed, not held, so a trace sees the uncoded state there.
- `--cut B:dq2,...`: the forward codes only the update's effect h - h_frozen at B; h_frozen, θ's residual for the same
  tokens, comes from frozen copies of the trained blocks (the receiving side recomputes it from the token's data), and
  its products are counted and declared as the code's. `dnq2` at boundary L (the final norm's input) codes
  rms_unit(h) - rms_unit(h_frozen) instead, which the residual's growth under training can't inflate, and hands on
  rms_unit(h_frozen) plus the code, which the scale-invariant final norm takes as it takes h. `dq2p64` hands on only 64 of the 512 rotated coordinates of
  the effect (a fixed subspace, both ways: 64·2 + 16 bits forward); `--cut-leak λ` adds λ times the share it leaves
  out to the loss, so the uncoded model the evaluator scores stays near the coded one (a statistic of the uncoded
  effect, so a trace then sees the uncoded state). `--confine` instead projects each trained wo and w_down's update
  onto the cut's span after every step, which puts the whole effect in the span when everything trained is in the
  block below the cut and that block is new (`--grow n`: n zero-output top blocks, copies of θ's with `--grow-copy`, E
  the grown model) or trains w_down alone; it refuses any other set (an existing block's trained wo reaches its MLP,
  whose frozen w_down maps that change out of the span).
- `--lora-rank r --lora-fix B --z-bits q`: a rank-r update whose B is r seeded orthonormal columns, so its whole effect
  on a token is r coefficients z = A·h, handed on (and their gradient back) at q bits each. `--lora-fix A` trains only
  B over a frozen random A.
- `--ia3` with `--set` naming wk, wv and w_up: IA3, a trained scale per output channel of each matrix (keys, values
  and MLP hidden units), E as encode_lora2's DoRA magnitudes over zero factors, which decode to exactly diag(l)·W.
- `--grow 1 --adapter r [--adapter-z q]`: a bottleneck adapter after the last block, as a grown block with zero
  attention and r live MLP hidden units (SwiGLU down to r, w_down back up, both trained, the step reading only their
  live slices), the r units optionally handed to w_down at q bits each (`--z-rotate`: in a randomized-DCT basis).
- `--prefix P` with a `--set` of every block's prefix_k, prefix_v and prefix_gate: prefix tuning, P learned key/value
  pairs in every block behind a zero per-head gate. `--block-adapter r` with a `--set` of every block's adapter_norm,
  adapter_down and adapter_up: Pfeiffer's bottleneck adapter of r units after every block's MLP, its up-projection from
  zero, its units optionally handed up at q bits each (`--block-adapter-z q`, rotated with `--z-rotate`). Both are model.py's extensions (`prefix`, `adapter`), which must be in the measurement commit; E is the model
  with them (`encode_model`).

Its source rides in the submission's notes. Every method marks explicit harness steps (`hooks.step`), so it runs in the
harness's script mode:

    python benchmarks/covert_witness/launch.py s50 --island wildcard --note "cachedhead ..." -- --set mlp-in:-1 --method cached
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

MAX_E_BYTES = 0
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


#: The optimal uniform quantizer's step for a unit Gaussian (Max 1960); rotated coordinates are near-Gaussian.
GAUSS_STEP = {1: 1.596, 2: 0.9957, 3: 0.5860, 4: 0.3352, 5: 0.1881, 6: 0.1041, 7: 0.0569, 8: 0.0308}
#: The widths whose plain code scales by GAUSS_STEP; wider plain codes scale by the absmax (an `m` code: every width).
GAUSS_PLAIN = (1, 2, 3)
#: The step minimizing a unit Gaussian's mean squared error under stochastic rounding onto the mid-rise grid, clipped
#: at the end levels (numerically, 4M draws): 0.18 at 2 bits against the absmax grid's 0.82 for a 512-vector.
STOCH_STEP = {1: 1.289, 2: 0.8409, 3: 0.5274, 4: 0.3085, 5: 0.1742, 6: 0.0996, 7: 0.0548, 8: 0.0299}


def quantize(y: torch.Tensor, bits: int, group: int, stochastic: bool, gauss: bool = False) -> torch.Tensor:
    """Each group of `group` trailing coordinates on a grid of 2^bits levels with one bf16 scale per group. Nearest
    rounding (forward) uses the mid-rise grid (k + 1/2)·step, step GAUSS_STEP times the group's RMS at GAUSS_PLAIN's
    widths (every width in GAUSS_STEP with `gauss`) and 2·absmax / 2^bits above. Stochastic rounding (gradients) uses
    the grid whose end levels are ±absmax (rounded up), so no value is clipped and the code is unbiased; with `gauss`
    it uses the mid-rise grid at STOCH_STEP times the group's RMS, unbiased between the end levels and clipped beyond."""
    shape = y.shape
    y = y.reshape(-1, shape[-1] // group, group)
    if stochastic and gauss:
        step = (y.pow(2).mean(-1, keepdim=True).sqrt() * STOCH_STEP[bits]).to(torch.bfloat16).float().clamp_min(1e-30)
        half = 2 ** (bits - 1)
        j = torch.floor(y / step - 0.5 + torch.rand_like(y)).clamp(-half, half - 1)
        return ((j + 0.5) * step).reshape(shape)
    if stochastic:
        top = (y.abs().amax(-1, keepdim=True) * (1 + 2 ** -7)).to(torch.bfloat16).float().clamp_min(1e-30)
        step = 2 * top / (2 ** bits - 1)
        j = torch.floor((y + top) / step + torch.rand_like(y)).clamp(0, 2 ** bits - 1)
        return (j * step - top).reshape(shape)
    if bits in GAUSS_PLAIN or gauss and bits in GAUSS_STEP:
        step = y.pow(2).mean(-1, keepdim=True).sqrt() * GAUSS_STEP[bits]
    else:
        step = y.abs().amax(-1, keepdim=True) * (2.0 / 2 ** bits)
    step = step.to(torch.bfloat16).float().clamp_min(1e-30)
    half = 2 ** (bits - 1)
    j = torch.floor(y / step).clamp(-half, half - 1)
    if bits <= 8 and group % 8 == 0:
        j = through_bytes((j + half).long(), bits).float() - half
    return ((j + 0.5) * step).reshape(shape)


def through_bytes(c: torch.Tensor, bits: int) -> torch.Tensor:
    """Codes c in [0, 2^bits) (trailing axis a multiple of 8) through one uint8 tensor, eight codes to `bits` bytes, and
    back unchanged: the code exists in the step at its own width, so a trace's narrowest cut (te_core 4.x's measured a)
    can be the code, where a float of the same values would count 32 bits a coordinate."""
    lanes = torch.arange(8, device=c.device, dtype=torch.int64)
    octets = torch.arange(bits, device=c.device, dtype=torch.int64)
    word = (c.reshape(*c.shape[:-1], -1, 8) << (bits * lanes)).sum(-1)
    packed = ((word.unsqueeze(-1) >> (8 * octets)) & 255).to(torch.uint8)
    word = (packed.long() << (8 * octets)).sum(-1)
    return ((word.unsqueeze(-1) >> (bits * lanes)) & (2 ** bits - 1)).reshape(c.shape)


#: Each `m` boundary's running mean of the coded residual (bf16 values in fp32), from the training batches coded before.
CUT_MEAN: dict = {}
MEAN_DECAY = 0.9


def code_bits(d: int, bits: int, group: int) -> int:
    """Bits per token of one coded d-vector: `bits` per coordinate and a bf16 step per group."""
    return d * bits + 16 * (d // group)


#: Entropy-coded codes (`e`): a token's rotated coordinates over its bf16 RMS are rounded onto step·j for any integer
#: j (nearest forward; stochastic back, so the gradient is unbiased and nothing is clipped), and the symbols are coded
#: under a model of |j| (E_BINS bins, the last an escape followed by an Elias-gamma count, a sign bit for j != 0) in
#: at most d·bits bits, an arithmetic coder's 2 included. A token takes the finest step step_0·2^(m/4), m < E_LEVELS,
#: that fits, else the fixed-rate code at the same width. The model is the EMA of earlier batches' symbols (16-bit
#: probabilities, charged once a session); a token's side bits are the RMS, the level flag and its share of the model.
E_BINS = 48
E_LEVELS = 8
E_DECAY = 0.9
#: A stochastic token takes the finest step whose expected length plus E_MARGIN standard deviations of its draw fits,
#: a choice its values alone make, so its rounding stays unbiased; a drawn length that still overflows (rare) takes
#: the fixed-rate code.
E_MARGIN = 4.0
ECODE: dict = {}


def e_side_bits() -> int:
    return 16 + math.ceil(math.log2(E_LEVELS + 1)) + math.ceil(16 * E_LEVELS * E_BINS / SESSION)


def symbol_probs(step: float, stochastic: bool) -> np.ndarray:
    """P(|j| in each bin) for a unit Gaussian rounded onto step·j, nearest (mid-tread) or stochastically. The code
    tables are numpy, so a re-trace on fake tensors (te_core.capture.script) builds the same ones."""
    x = np.linspace(0, 12, 240001)
    w = np.exp(-x ** 2 / 2)
    w = w / w.sum()
    u = x / step
    if stochastic:
        f = np.floor(u)
        r = u - f
        return (np.bincount(np.minimum(f, E_BINS - 1).astype(np.int64), w * (1 - r), E_BINS)
                + np.bincount(np.minimum(f + 1, E_BINS - 1).astype(np.int64), w * r, E_BINS))
    return np.bincount(np.minimum(np.floor(u + 0.5), E_BINS - 1).astype(np.int64), w, E_BINS)


def bin_bits(p):
    """Code lengths per |j| bin under model p (a tensor or an array), held as 16-bit probabilities (each at least
    2^-16), sign included."""
    if isinstance(p, np.ndarray):
        q = np.maximum(np.round(p / p.sum() * 2 ** 16), 1)
        return -np.log2(q / q.sum()) + (np.arange(len(p)) > 0)
    q = (p / p.sum() * 2 ** 16).round().clamp_min(1)
    return -torch.log2(q / q.sum()) + (torch.arange(len(p), device=p.device) > 0).to(p.dtype)


def counts(idx: torch.Tensor, n: int) -> torch.Tensor:
    """torch.bincount(idx, minlength=n) for idx < n, at a fixed size (a fake tensor's bincount has none)."""
    idx = idx.flatten()
    return torch.zeros(n, dtype=torch.long, device=idx.device).scatter_add_(0, idx, torch.ones_like(idx))


def symbol_bits(j: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
    top = E_BINS - 1
    a = j.abs()
    n = (a - top + 1).clamp_min(1)
    return lb[a.clamp(max=top).long()] + torch.where(a >= top, 2 * torch.floor(torch.log2(n)) + 1, 0.0)


def e_step(d: int, bits: int, stochastic: bool) -> float:
    """The finest step at which a Gaussian token's mean length plus one standard deviation, and the coder's 2 bits,
    fit d·bits."""
    lo, hi = 1e-3, 16.0
    for _ in range(50):
        s = (lo * hi) ** 0.5
        p = symbol_probs(s, stochastic)
        lb = bin_bits(p)
        mean = float((p * lb).sum())
        var = max(0.0, float((p * lb ** 2).sum()) - mean ** 2)
        lo, hi = (lo, s) if d * mean + (d * var) ** 0.5 + 2 <= d * bits else (s, hi)
    return hi


def ecode(y: torch.Tensor, bits: int, key, stochastic: bool, learn: bool) -> torch.Tensor:
    """Each token (a row of y) in the entropy-coded code of ECODE[key] (E_BINS above); with `learn` the token's
    symbols then move the model (so a token's code reads only earlier batches)."""
    shape, d = y.shape, y.shape[-1]
    y = y.reshape(-1, d)
    st = ECODE.get(key)
    if st is None:
        steps = [e_step(d, bits, stochastic) * 2 ** (m / 4) for m in range(E_LEVELS)]
        p = torch.tensor(np.stack([symbol_probs(s, stochastic) for s in steps]), dtype=torch.float32, device=y.device)
        st = ECODE[key] = {"steps": steps, "p": p, "lb": torch.stack([bin_bits(q) for q in p]),
                           "count": torch.zeros(E_LEVELS + 2, device=y.device), "length": torch.zeros((), device=y.device)}
    rms = y.pow(2).mean(-1, keepdim=True).sqrt().to(torch.bfloat16).float().clamp_min(1e-30)
    w = y / rms
    budget = d * bits - 2
    level = torch.full((len(y),), E_LEVELS, dtype=torch.long, device=y.device)
    length = torch.zeros(len(y), device=y.device)
    sym = torch.zeros_like(w)
    for m in reversed(range(E_LEVELS)):
        u = w / st["steps"][m]
        lb = st["lb"][m]
        if stochastic:
            f = torch.floor(u)
            r = u - f
            l0, l1 = symbol_bits(f, lb), symbol_bits(f + 1, lb)
            spread = (r * (1 - r) * (l1 - l0) ** 2).sum(-1).sqrt()
            fits = ((1 - r) * l0 + r * l1).sum(-1) + E_MARGIN * spread <= budget
            j = f + (torch.rand_like(r) < r).float()
        else:
            j = torch.floor(u + 0.5)
        n = symbol_bits(j, lb).sum(-1)
        if not stochastic:
            fits = n <= budget
        level = torch.where(fits, m, level)
        length = torch.where(fits, n, length)
        sym = torch.where(fits[:, None], j, sym)
    coded = level < E_LEVELS
    drawn_over = coded & (length > budget)
    coded &= ~drawn_over
    step = torch.tensor(st["steps"], device=y.device)[level.clamp(max=E_LEVELS - 1)]
    out = torch.where(coded[:, None], sym * step[:, None] * rms, quantize(y, bits, d, stochastic, not stochastic))
    if learn:
        with torch.no_grad():
            pair = torch.where(coded, level, E_LEVELS)[:, None] * E_BINS + sym.abs().clamp(max=E_BINS - 1).long()
            hists = counts(pair, (E_LEVELS + 1) * E_BINS).reshape(E_LEVELS + 1, E_BINS)
            for m in range(E_LEVELS):
                hist = hists[m]
                if hist.sum() > 0:
                    st["p"][m] = E_DECAY * st["p"][m] + (1 - E_DECAY) * hist.float() / hist.sum()
                    st["lb"][m] = bin_bits(st["p"][m])
            st["count"] += counts(torch.where(drawn_over, E_LEVELS + 1, level), E_LEVELS + 2)
            st["length"] += torch.where(coded, length, 0.0).sum()
    return out.reshape(shape)


def ecode_report() -> dict:
    """Per e code: its steps (in units of the token's RMS), how many tokens took each step, the fixed-rate code
    (`fixed`) and a drawn overflow (`overflow`), and the mean payload bits of an entropy-coded token."""
    out = {}
    for key, st in ECODE.items():
        c = [int(x) for x in st["count"].tolist()]
        out[f"{key[0]}{key[1]}"] = {"steps": [round(s, 4) for s in st["steps"]], "tokens_by_step": c[:E_LEVELS],
                                    "fixed": c[E_LEVELS], "overflow": c[E_LEVELS + 1],
                                    "mean_bits": float(st["length"]) / max(1, sum(c[:E_LEVELS]))}
    return out


#: Rice-coded forward codes (`r`), which a traced step holds at their width: a token's rotated coordinates over its
#: bf16 RMS are rounded onto step·j for any integer j (nearest), zigzagged (z = 2|j| - [j < 0]) and Rice-coded with
#: parameter k (z's k low bits at fixed offsets, then every z >> k in unary) in one d·bits-bit field held as bytes. A
#: token takes the finest step step_0·2^(m/4), m < E_LEVELS, whose code fits (step_0 and each step's k set on a unit
#: Gaussian), else the fixed-rate Gaussian grid at the same width in the same bytes. Its level (uint8) and RMS (bf16)
#: travel beside the bytes, and the decoder reads only those three tensors, so a = d·bits + 24.
RCODE: dict = {}
R_TABLES: dict = {}
R_SIDE_BITS = 16 + 8


def zigzag_rice_bits(j: torch.Tensor, k) -> torch.Tensor:
    return k + torch.floor((2 * j.abs() - (j < 0).to(j.dtype)) / 2 ** k) + 1


def r_tables(d: int, bits: int) -> dict:
    """Each level's step (in units of the token's RMS) and Rice parameter: step_0 is the finest step at which a
    unit-Gaussian token's mean code length plus one standard deviation fits d·bits under its best parameter, and each
    level takes the parameter with the least mean length at its step. Numpy, like symbol_probs."""
    erf = np.vectorize(math.erf)

    def stats(s: float, k: int) -> tuple[float, float]:
        j = np.arange(-int(9 / s) - 2, int(9 / s) + 3, dtype=np.float64)
        p = 0.5 * (erf((j + 0.5) * s / 2 ** 0.5) - erf((j - 0.5) * s / 2 ** 0.5))
        n = k + np.floor((2 * np.abs(j) - (j < 0)) / 2 ** k) + 1
        mean = float((p * n).sum())
        return mean, max(0.0, float((p * n * n).sum()) - mean ** 2)

    best = None
    for k in range(8):
        lo, hi = 1e-3, 16.0
        for _ in range(50):
            s = (lo * hi) ** 0.5
            mean, var = stats(s, k)
            lo, hi = (lo, s) if d * mean + (d * var) ** 0.5 <= d * bits else (s, hi)
        best = (hi, k) if best is None else min(best, (hi, k))
    steps = [best[0] * 2 ** (m / 4) for m in range(E_LEVELS)]
    return {"steps": steps, "k": [min(range(8), key=lambda k: stats(s, k)[0]) for s in steps]}


def rcode(y: torch.Tensor, bits: int, key, count: bool = False) -> torch.Tensor:
    """Each token (a row of y) through the Rice code above: encoded to bytes, a level and an RMS, then decoded from
    those alone. With `count`, RCODE[key] counts the tokens at each level and the coded tokens' bits, read on the
    decoded side."""
    shape, d = y.shape, y.shape[-1]
    y = y.reshape(-1, d)
    T, n, dev = len(y), d * bits, y.device
    tab = R_TABLES.get((d, bits))
    if tab is None:
        tab = R_TABLES[(d, bits)] = r_tables(d, bits)
    steps, ks = torch.tensor(tab["steps"], device=dev), torch.tensor(tab["k"], device=dev)
    half, gstep = 2 ** (bits - 1), GAUSS_STEP[bits]
    i = torch.arange(d, device=dev)
    lanes = torch.arange(8, device=dev)
    kmax = max(max(tab["k"]), bits)

    rms = y.pow(2).mean(-1, keepdim=True).sqrt().to(torch.bfloat16)
    w = y / rms.float().clamp_min(1e-30)
    level = torch.full((T,), E_LEVELS, dtype=torch.long, device=dev)
    z = torch.floor(w / gstep).clamp(-half, half - 1).long() + half
    for m in reversed(range(E_LEVELS)):
        jm = torch.floor(w / tab["steps"][m] + 0.5)
        fits = zigzag_rice_bits(jm, tab["k"][m]).sum(-1) <= n
        level = torch.where(fits, m, level)
        z = torch.where(fits[:, None], (2 * jm.abs() - (jm < 0).float()).long(), z)
    k = torch.where(level == E_LEVELS, bits, ks[level.clamp(max=E_LEVELS - 1)])[:, None]
    pos = torch.arange(n, device=dev)
    ends = d * k + torch.cumsum((z >> k) + 1, -1) - 1
    k1 = k.clamp_min(1)
    low = (z.gather(1, (pos // k1).clamp(max=d - 1)) >> (pos % k1)) & 1
    nxt = torch.searchsorted(ends, pos.expand(T, n).contiguous()).clamp(max=d - 1)
    field = torch.where(pos < d * k, low, (ends.gather(1, nxt) == pos).long())
    packed = (field.reshape(T, n // 8, 8) << lanes).sum(-1).to(torch.uint8)
    lev = level.to(torch.uint8)

    bit = ((packed.long().unsqueeze(-1) >> lanes) & 1).reshape(T, n)
    L = lev.long()
    fixed = L == E_LEVELS
    k = torch.where(fixed, bits, ks[L.clamp(max=E_LEVELS - 1)])[:, None]
    low = torch.zeros(T, d, dtype=torch.long, device=dev)
    for b in range(kmax):
        on = b < k
        low = low | ((bit.gather(1, torch.where(on, i * k + b, 0)) & on.long()) << b)
    cs = torch.cumsum((bit == 1) & (pos >= d * k), -1)
    at = torch.searchsorted(cs, (i + 1).expand(T, d).contiguous()).clamp(max=n - 1)
    q = at - torch.cat([d * k - 1, at[:, :-1]], -1) - 1
    zz = (q << k) | low
    j = torch.where((zz & 1).bool(), -(zz + 1) // 2, zz // 2).float()
    r = rms.float()
    out = torch.where(fixed[:, None], ((low - half).float() + 0.5) * gstep * r,
                      j * steps[L.clamp(max=E_LEVELS - 1)][:, None] * r)
    if count:
        st = RCODE.setdefault(key, {"count": torch.zeros(E_LEVELS + 1, device=dev), "length": torch.zeros((), device=dev)})
        st["count"] += counts(L, E_LEVELS + 1)
        st["length"] += torch.where(fixed, 0, at[:, -1] + 1).sum()
    return out.reshape(shape)


def rcode_report() -> dict:
    """Per r code: its steps (in units of the token's RMS) and Rice parameters, how many tokens took each step and the
    fixed-rate code (`fixed`), and the mean bits of a Rice-coded token."""
    out = {}
    for key, st in RCODE.items():
        c = [int(x) for x in st["count"].tolist()]
        out[f"{key[0]}{key[1]}"] = {"tokens_by_step": c[:E_LEVELS], "fixed": c[E_LEVELS],
                                    "mean_bits": float(st["length"]) / max(1, sum(c[:E_LEVELS]))}
    out["tables"] = {f"{d}x{b}": {"steps": [round(s, 4) for s in t["steps"]], "k": t["k"]}
                     for (d, b), t in R_TABLES.items()}
    return out


class Coded(torch.autograd.Function):
    """A token's d-vector handed across a cut as a code: rotated by R, quantized per group, rotated back (forward,
    nearest rounding); the backward passes the gradient straight through the quantizer and hands it across the cut in
    the same code with stochastic rounding. With `rotate` false the vector is coded in its own basis. With `outliers`
    k > 0 the forward first takes out each token's k largest coordinates, which go apart at bf16 (with their indices),
    and codes the rest. With a `mean` (d,) the forward codes x − mean on the GAUSS_STEP grid at every width and adds the
    mean back. With `grad_gauss` the gradient goes on quantize's RMS-scaled stochastic grid (STOCH_STEP). An `ekeys`
    entry (forward, back) codes that direction in the entropy-coded code of ECODE[key] (`ecode`); the forward's model
    learns only with `learn`. A forward key ending in "r" codes the forward in the Rice code (`rcode`). With `keep`
    only R's first keep rows code (both ways), so the vector is projected onto their span; `group` is then keep."""

    @staticmethod
    def forward(ctx, x, bits_f, bits_b, group, seed, rotate, outliers=0, mean=None, grad_gauss=False, ekeys=(None, None),
                learn=False, keep=0):
        ctx.cfg = (bits_b, group, seed, rotate, grad_gauss, ekeys[1], keep)
        with torch.autocast(x.device.type, enabled=False):
            y = x.float() if mean is None else x.float() - mean
            if outliers:
                idx = y.abs().topk(outliers, dim=-1).indices
                val = y.gather(-1, idx).to(torch.bfloat16).float()
                y = y.scatter(-1, idx, 0.0)
            if rotate:
                r = fresh_rotation(x.shape[-1], seed, x.device)
                r = r[:keep] if keep else r
                z = y @ r.T
                if ekeys[0] and ekeys[0][1] == "r":
                    z = rcode(z, bits_f, ekeys[0], learn)
                elif ekeys[0]:
                    z = ecode(z, bits_f, ekeys[0], False, learn)
                else:
                    z = quantize(z, bits_f, group, False, mean is not None)
                y = z @ r
            else:
                y = quantize(y, bits_f, group, False, mean is not None)
            if outliers:
                y = y.scatter(-1, idx, val)
            return (y if mean is None else y + mean).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        bits_b, group, seed, rotate, grad_gauss, ekey, keep = ctx.cfg
        with torch.autocast(g.device.type, enabled=False):
            y = g.float()
            if rotate:
                r = fresh_rotation(g.shape[-1], seed, g.device)
                r = r[:keep] if keep else r
                z = y @ r.T
                out = (ecode(z, bits_b, ekey, True, True) if ekey else quantize(z, bits_b, group, True, grad_gauss)) @ r
            else:
                out = quantize(y, bits_b, group, True, grad_gauss)
        return out.to(g.dtype), None, None, None, None, None, None, None, None, None, None, None


CUT_RE = re.compile(r"(?P<delta>d?)(?P<norm>n?)q(?P<bf>\d+)(?P<mean>m?)(?P<fe>[er]?)(?:p(?P<keep>\d+))?"
                    r"(?:o(?P<k>\d+))?(?:b(?P<bb>\d+)(?P<bs>[se]?))?(?:g(?P<group>\d+))?")


def rms_unit(x: torch.Tensor) -> torch.Tensor:
    """x over its RMS per token, which the final RMSNorm maps to the same output as x (it is scale-invariant)."""
    y = x.float()
    return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-12)).to(x.dtype)


def parse_cut(spec: str, L: int, d: int, low: int | None = None) -> list[dict]:
    """`B:CODE,...`: boundary B (1..L; the input of block B, L the input of the final norm; `all` every one) handed on
    as CODE = [d][n]q<forward bits>[m][e|r][p<keep>][o<k>][b<backward bits>[s|e]][g<group>] in a randomized-DCT basis
    (group d unless given). `n` (boundary L only) hands on the state over its RMS per token (`rms_unit`), which the
    final RMSNorm maps to the same output, so the residual's growth under training can't inflate what is coded; with
    `d` that is rms_unit(h) - rms_unit(h_frozen). `p<keep>` (a `d` code only) hands on only the first `keep` rotated coordinates of
    h - h_frozen, a fixed subspace, both ways, so a crossing is keep·bits + side bits; the effect outside it never
    crosses (`--cut-leak` penalizes it, since the evaluator scores the uncoded model). `o<k>` sends each token's k
    largest coordinates apart, a bf16 value and a ceil(log2 d)-bit index
    each. `m` codes the forward about the boundary's running mean (CUT_MEAN; earlier batches' coded states only, so
    causal and computed on the receiving side, no bits sent) on the Gaussian grid. `b<bits>s` rounds the gradient onto the
    RMS-scaled grid (STOCH_STEP) at the same bits. `e` (after q's bits, or b's) codes that direction entropy-coded in
    d·bits bits plus e_side_bits (`ecode`; whole tokens, no o<k>); no coder runs, so a traced step holds a forward `e`
    code at the uncoded state's width. `r` (after q's bits) codes the forward in the Rice code, d·bits bits plus
    R_SIDE_BITS, which the step holds as bytes (`rcode`; whole tokens, no o<k>, d a multiple of 8, 2-8 bits). With the `d` prefix
    the forward codes only the update's effect h - h_frozen, where h_frozen is θ's residual at
    B for the same tokens, which the receiving side recomputes from the token's data; it needs B above the lowest
    trained block `low` (at or below it the effect is 0)."""
    out = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        where, code = part.split(":")
        m = CUT_RE.fullmatch(code)
        if not m:
            raise SystemExit(f"wildcard: --cut code {code!r} is not "
                             f"[d][n]q<bits>[m][e|r][p<keep>][o<k>][b<bits>[s|e]][g<group>]")
        delta = m.group("delta") == "d"
        norm = m.group("norm") == "n"
        if norm and (where != str(L) or m.group("mean")):
            raise SystemExit(f"wildcard: --cut {part!r}: `n` hands on the state over its RMS, the same to the final "
                             f"norm only, so it needs boundary L = {L} (and no `m`)")
        mean = m.group("mean") == "m"
        bf = int(m.group("bf"))
        keep = int(m.group("keep") or 0)
        k = int(m.group("k") or 0)
        bb = int(m.group("bb") or bf)
        group = int(m.group("group") or d)
        if d % group or not (1 <= bf <= 16 and 1 <= bb <= 16) or not 0 <= k < d:
            raise SystemExit(f"wildcard: --cut {part!r}: bits 1-16, outliers below d and a group dividing d = {d}")
        grad_gauss = m.group("bs") == "s"
        if mean and (delta or k or bf not in GAUSS_STEP):
            raise SystemExit(f"wildcard: --cut {part!r}: `m` codes a whole state (no d prefix), without o<k>, at "
                             f"{min(GAUSS_STEP)}-{max(GAUSS_STEP)} bits")
        if grad_gauss and bb not in STOCH_STEP:
            raise SystemExit(f"wildcard: --cut {part!r}: `b<bits>s` at {min(STOCH_STEP)}-{max(STOCH_STEP)} bits")
        ecode_f, ecode_b, rcode_f = m.group("fe") == "e", m.group("bs") == "e", m.group("fe") == "r"
        if (ecode_f or ecode_b or rcode_f) and (group != d or ((ecode_f or rcode_f) and k) or not (bf <= 8 and bb <= 8)):
            raise SystemExit(f"wildcard: --cut {part!r}: an `e` or `r` code takes whole tokens (no g<group>), no o<k>, "
                             f"at most 8 bits")
        if keep and (not delta or k or m.group("group") or keep % 8 or not 8 <= keep < d):
            raise SystemExit(f"wildcard: --cut {part!r}: `p<keep>` keeps a `d` code's first keep rotated coordinates "
                             f"(no o<k> or g<group>), keep a multiple of 8 from 8 to below d = {d}")
        if rcode_f and (d % 8 or bf < 2):
            raise SystemExit(f"wildcard: --cut {part!r}: an `r` code needs d = {d} a multiple of 8 and at least 2 bits "
                             f"(at 1 bit a Gaussian token's every coordinate rounds to 0)")
        n = keep or d
        group = keep or group
        side = k * (16 + math.ceil(math.log2(d)))
        fwd = (n * bf + e_side_bits() if ecode_f else n * bf + R_SIDE_BITS if rcode_f else code_bits(n, bf, group)) + side
        bwd = n * bb + e_side_bits() if ecode_b else code_bits(n, bb, group)
        for b in (range(1, L + 1) if where == "all" else [int(where)]):
            if not 1 <= b <= L:
                raise SystemExit(f"wildcard: --cut boundary {b} is not in 1..{L}")
            if delta and (low is None or not 0 <= low < b):
                raise SystemExit(f"wildcard: --cut {b}:{code} codes h - h_frozen, which needs trained blocks only, "
                                 f"the lowest below boundary {b} (lowest trained block: {low})")
            out.append({"boundary": b, "bits_fwd": bf, "bits_bwd": bb, "group": group, "delta": delta, "norm": norm,
                        "outliers": k,
                        "mean": mean, "grad_gauss": grad_gauss, "ecode_fwd": ecode_f, "ecode_bwd": ecode_b,
                        "rcode_fwd": rcode_f, "keep": keep, "a_fwd_bits": fwd, "a_bwd_bits": bwd})
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


#: With --grow: the grown model's config and the start's block count (blocks from it on are new).
GROWN: dict = {}


def grow(net, rec: dict, theta: dict, n: int, seed: int, adapter: int = 0, copy: bool = False) -> tuple[dict, dict]:
    """θ's model grown in place by n new top blocks, each in model.py's own init (`copy`: new block i a copy of θ's block
    ⌊iL/n⌋, stacking growth) with its output matrices (wo, w_down) zeroed in place, so the grown model computes θ's
    function and θ's tensors are never rewritten: (the grown model's record, its starting tensors, θ's and the new
    blocks'). With `adapter` r the new block is a bottleneck adapter: its attention's matrices are all zero (so it adds
    nothing) and only the first r of its MLP's hidden units are live (w_gate's and w_up's other rows zero, and
    bottleneck runs it on the live slices alone)."""
    from model import Block

    L = rec["config"]["n_layer"]
    torch.manual_seed(seed)
    with torch.device(prepare.DEVICE):
        new = [Block(net.inner.size, 0.02, 0.02) for _ in range(n)]
    with torch.no_grad():
        for i, blk in enumerate(new):
            if copy:
                src = f"blocks.{i * L // n}."
                for k, p in blk.named_parameters():
                    p.copy_(theta[src + k])
            blk.wo.weight.zero_()
            blk.w_down.weight.zero_()
            if adapter:
                for m in (blk.wq, blk.wk, blk.wv):
                    m.weight.zero_()
                blk.w_gate.weight[adapter:].zero_()
                blk.w_up.weight[adapter:].zero_()
    net.inner.blocks.extend(new)
    config = net.config = {**rec["config"], "n_layer": L + n}
    start = {**theta, **{k: p.detach().cpu().clone() for k, p in net.params().items() if k not in theta}}
    GROWN.update(config=config, start=L, adapter=adapter, new=sorted(k for k in start if k not in theta))
    return {**rec, **prepare.model_record(rec, config, start)}, start


#: model.py's extensions (cursor/te-prefix-82af): a config key, and the modules it adds to every block.
EXTENSIONS = {"prefix": ("prefix_k", "prefix_v", "prefix_gate"), "adapter": ("adapter_norm", "adapter_down", "adapter_up")}


def extend(net, rec: dict, theta: dict, key: str, n: int, seed: int) -> tuple[dict, dict]:
    """θ's model given model.py's extension `key` n in place, every block's new modules taken from a model.py Block built
    with it (its own init: a zero prefix gate, a zero adapter up-projection), so the model computes θ's function; as
    grow, (the record, its starting tensors)."""
    import inspect

    from model import LM, Block

    if key not in inspect.signature(LM).parameters:
        raise SystemExit(f"wildcard: this needs model.py's `{key}` in the measurement commit (cursor/te-prefix-82af)")
    size, L = net.inner.size, rec["config"]["n_layer"]
    torch.manual_seed(seed)
    with torch.device(prepare.DEVICE):
        for blk in net.inner.blocks:
            new = Block(size, 0.02, 0.02 / math.sqrt(2 * L), **{key: n})
            for name in EXTENSIONS[key]:
                setattr(blk, name, getattr(new, name))
    config = net.config = {**rec["config"], key: n}
    start = {**theta, **{k: p.detach().cpu().clone() for k, p in net.params().items() if k not in theta}}
    GROWN.update(config=config, start=rec["config"]["n_layer"], adapter=0,
                 new=sorted(k for k in start if k not in theta))
    return {**rec, **prepare.model_record(rec, config, start)}, start


def frozen_flops(rec: dict, lo: int, hi: int) -> int:
    """Forward FLOPs per token of blocks lo..hi-1."""
    names = tuple(f"blocks.{j}." for j in range(lo, hi))
    mm = sum(n["flops"] for n in prepare.positions(rec) if n["param"].startswith(names))
    return mm + (hi - lo) * prepare.attention_flops(rec, prepare.EVAL_CTX)


def install_cuts(net, cuts: list[dict], theta: dict | None = None, low: int | None = None, grown_from: int = 10 ** 6):
    """Each boundary's code as a forward hook on the block below it (boundary B: block B - 1's output), active while
    CUT["on"]. A delta code's hook adds h_frozen back after coding h - h_frozen; h_frozen is carried up from block
    `low`'s input through frozen copies of blocks low.. (frozen_blocks), except blocks from `grown_from` on, new blocks
    whose θ outputs are zero, which pass it through. An `m` code's hook codes about CUT_MEAN, then
    moves CUT_MEAN toward this batch's coded mean only in a training forward or with CUT["calibrate"] (never a
    probe's), so a token's code reads no later token, and the mean reads nothing the receiving side lacks. An `e`
    code's model learns on the same terms (`ecode`). While CUT["leak"] is a list, a `p<keep>` code's training forward
    appends the mean square of h - h_frozen outside its kept span over h_frozen's (a term the trained side computes
    from its own values, which train_iterative adds to the loss)."""
    CUT_MEAN.clear()
    ECODE.clear()
    RCODE.clear()
    by_block = {c["boundary"] - 1: c for c in cuts}
    top = max((c["boundary"] for c in cuts if c["delta"]), default=None)
    carried = set(range(low, top)) if top is not None else set()
    frozen = frozen_blocks(net, theta, low, min(top, grown_from)) if top is not None and low < grown_from else {}
    state = {}
    handles = []
    for j in sorted(set(by_block) | carried):
        def hook(m, inp, out, j=j, c=by_block.get(j)):
            if not CUT["on"]:
                return out
            f = None
            if j in carried:
                with torch.no_grad():
                    below = state.pop(j) if j > low else inp[0].detach()
                    f = state[j + 1] = frozen[j](below, inp[1], inp[2]) if j in frozen else below
            if c is None:
                return out
            if c["norm"]:
                out = rms_unit(out)
                f = rms_unit(f) if f is not None else None
            seed = 1000 + c["boundary"]
            learn = torch.is_grad_enabled() or bool(CUT.get("calibrate"))
            fkey = (c["boundary"], "f") if c["ecode_fwd"] else (c["boundary"], "r") if c["rcode_fwd"] else None
            ekeys = (fkey, (c["boundary"], "b") if c["ecode_bwd"] else None)
            if CUT.get("spectrum") is not None:
                with torch.no_grad():
                    of = ("n(h) - n(h_frozen)" if c["norm"] else "h - h_frozen") if c["delta"] else \
                        "n(h)" if c["norm"] else "h"
                    acc = CUT["spectrum"].setdefault(c["boundary"], {"of": of})
                    spectrum_add(acc, out - f if c["delta"] else out, out)
                    if c["keep"]:
                        acc["outside_sq"] = acc.get("outside_sq", 0.0) + float(outside(out - f, c["keep"], seed)
                                                                               .pow(2).sum())
            if c["delta"]:
                if c["keep"] and CUT.get("leak") is not None and torch.is_grad_enabled():
                    CUT["leak"].append(outside(out - f, c["keep"], seed).pow(2).mean()
                                       / f.detach().float().pow(2).mean().clamp_min(1e-30))
                return f + Coded.apply(out - f, c["bits_fwd"], c["bits_bwd"], c["group"], seed, True, c["outliers"],
                                       None, c["grad_gauss"], ekeys, learn, c["keep"])
            mu = None
            if c["mean"]:
                mu = CUT_MEAN.get(c["boundary"])
                mu = torch.zeros(out.shape[-1], device=out.device) if mu is None else mu
            coded = Coded.apply(out, c["bits_fwd"], c["bits_bwd"], c["group"], seed, True, c["outliers"], mu,
                                c["grad_gauss"], ekeys, learn)
            if c["mean"] and learn:
                with torch.no_grad():
                    now = coded.detach().float().mean(tuple(range(coded.dim() - 1)))
                    old = CUT_MEAN.get(c["boundary"])
                    new = now if old is None else MEAN_DECAY * old + (1 - MEAN_DECAY) * now
                    CUT_MEAN[c["boundary"]] = new.to(torch.bfloat16).float()
            return coded
        handles.append(net.inner.blocks[j].register_forward_hook(hook))
    return handles


def outside(y: torch.Tensor, keep: int, seed: int) -> torch.Tensor:
    """The part of y outside the span of the rotation's first keep rows (what a `p<keep>` cut never hands on)."""
    with torch.autocast(y.device.type, enabled=False):
        r = fresh_rotation(y.shape[-1], seed, y.device)[:keep]
        y = y.float()
        return y - (y @ r.T) @ r


SESSION = 4096
RANKS = (4, 8, 16, 32, 64, 128)
COORDS = (1, 2, 4, 8, 16, 32)


def spectrum_add(acc: dict, delta: torch.Tensor, h: torch.Tensor):
    """Accumulate, per SESSION-token session of `delta` (windows, ctx, d; δ = h for a whole-state code): the share of
    its energy that its top r directions hold (eigenvalues λ of δᵀδ), the transform-coding gain ½·log2(mean λ / geomean
    λ) (bits per coordinate a fixed basis with bits allocated by variance saves over a flat code at high rate) and the
    same over the coordinates' own variances (`channel_gain_bits`, the gain without a basis; `centered_gain_bits` with
    each coordinate's session mean taken out), the share of a session's
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
    peak = x.pow(2).sum(-1)
    acc["sessions"] = acc.get("sessions", 0) + share.shape[0]
    acc["gain_bits"] = acc.get("gain_bits", 0.0) + float((-0.5 * torch.log2(lam).mean(-1)).sum())
    for key, var in (("channel_gain_bits", x.pow(2).mean(1)), ("centered_gain_bits", x.var(1))):
        var = (var / var.mean(-1, keepdim=True).clamp_min(1e-30)).clamp_min(1e-9)
        acc[key] = acc.get(key, 0.0) + float((-0.5 * torch.log2(var).mean(-1)).sum())
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
                       **({"outside_share": acc["outside_sq"] / max(acc["delta_sq"], 1e-30)} if "outside_sq" in acc
                          else {}),
                       **{k: v / s for k, v in acc.items()
                          if s and (k.startswith("top") or k.endswith("gain_bits") or k == "token_peak")},
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
    fits --max-e-bytes. `fix` "A" freezes that random A (only B trains). `fix` "B" makes B r seeded orthonormal
    columns (rows of a randomized DCT), frozen, with A = 0 trained and s = sqrt(d_out / r), so the update's whole effect
    on a token is its r coefficients z = A·h; `zbits` hands z on coded at that many bits per coefficient (one bf16 scale
    per token), and its gradient back the same way."""
    if rank < 0:
        cap = MAX_E_BYTES
        if not cap or fix:
            raise SystemExit("wildcard: --lora-rank -1 needs --max-e-bytes and no --lora-fix")
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


def add_ia3(net, theta: dict, params: dict, mats: list[str]) -> dict:
    """IA3 (Liu et al. 2022) on `mats`: each matrix's output channels times a trained fp32 scale l (from 1), W' =
    diag(l)·W; on wk, wv and w_up that is IA3's keys, values and MLP hidden units. θ's matrices stay frozen."""
    scales = {}
    for n in mats:
        params[n].requires_grad_(False)
        params[n].data = theta[n].to(params[n].device).clone()
        l = scales[n] = torch.nn.Parameter(torch.ones(theta[n].shape[0], device=prepare.DEVICE))

        def hook(m, inp, out, l=l):
            return (out.float() * l).to(out.dtype)
        net.inner.get_submodule(n.removesuffix(".weight")).register_forward_hook(hook)
    return scales


def encode_ia3(theta: dict, scales: dict, full: dict) -> bytes:
    """IA3's scales as CWLORA2 magnitudes: zero rank-1 factors, so lora_merge rescales θ's row i to norm l_i·|W_i|,
    which is l_i·W_i; magnitudes and the gains carried in full are stored at fp32."""
    fac, mags = {}, {}
    for n, l in scales.items():
        w = theta[n].float()
        fac[n] = (torch.zeros(1, w.shape[1]), torch.zeros(w.shape[0], 1), 1.0)
        mags[n] = l.detach().float().cpu() * torch.linalg.vector_norm(w, dim=1)
    return prepare.encode_lora2(fac, 8, "f32", mags, full)


#: The seed of --z-rotate's randomized DCT for the adapters' unit codes.
Z_ROTATE_SEED = 4099


def bottleneck(net, params: dict, tuned: list[str], r: int, zbits: int, method: str, rotate: bool = False) -> None:
    """Run a grown adapter block's MLP on its live parts alone (w_gate's and w_up's first r rows, w_down's first r
    columns), so the step casts and multiplies only those and the dead parts get no gradient (AdamW leaves a zero entry
    with a zero gradient at zero), and with `zbits` hand its r hidden units to w_down as a code (`rotate`: in a
    randomized-DCT basis)."""
    j = GROWN["start"]
    live = re.compile(rf"blocks\.{j}\.(mlp_norm|(w_gate|w_up|w_down)\.weight)")
    if method != "adamw" or GROWN["config"]["n_layer"] != j + 1 or not all(live.fullmatch(n) for n in tuned):
        raise SystemExit(f"wildcard: --adapter trains block {j}'s mlp_norm, w_gate, w_up and w_down (one grown block, "
                         f"AdamW, which leaves a zero entry with zero gradient at zero)")
    blk = net.inner.blocks[j]
    for m in (blk.w_gate, blk.w_up):
        m.forward = lambda x, m=m: F.linear(x, m.weight[:r])
    blk.w_down.forward = lambda x, m=blk.w_down: F.linear(x, m.weight[:, :r])
    if zbits:
        blk.w_down.register_forward_pre_hook(
            lambda m, inp: (Coded.apply(inp[0], zbits, zbits, r, Z_ROTATE_SEED, rotate, 0),))


def flop_budget(a) -> float:
    """--budget-fraction of the tier's training FLOPs (`prepare.budget`: the search budget, or the long tier's
    LONG_C0 x the start's C0 under CW_TIER=long)."""
    return a.budget_fraction * prepare.budget(a.start)["train_flops"]


def train_iterative(a, net, rec, theta, params, tuned, data, fwd=None):
    fwd = fwd or net
    mats = [n for n in tuned if params[n].ndim == 2]
    gains = [n for n in tuned if params[n].ndim == 1]
    opts, extra, factors, scales = [], 0, {}, {}
    if GROWN.get("adapter"):
        bottleneck(net, params, tuned, GROWN["adapter"], a.adapter_z, a.method, a.z_rotate)
    if a.block_adapter_z:
        for blk in net.inner.blocks:
            blk.adapter_up.register_forward_pre_hook(
                lambda m, inp: (Coded.apply(inp[0], a.block_adapter_z, a.block_adapter_z, a.block_adapter,
                                            Z_ROTATE_SEED, a.z_rotate, 0),))
    if a.ia3:
        if a.lora_rank or a.confine or GROWN or not mats:
            raise SystemExit("wildcard: --ia3 scales the output channels of --set's matrices of θ; it takes no "
                             "--lora-rank, --confine or --grow")
        scales = add_ia3(net, theta, params, mats)
        trainable = {"full": gains, "adapters": {n: 1 for n in mats}}
        groups = [{"params": list(scales.values()), "lr": a.lr, "weight_decay": 0.0},
                  {"params": [params[n] for n in gains], "lr": a.norm_lr}]
        opts.append((torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, a.beta2), weight_decay=a.wd,
                                       fused=prepare.DEVICE.startswith("cuda")), None))
    elif a.lora_rank:
        factors, a.lora_rank = add_adapters(net, theta, params, mats, gains, a.lora_rank, a.lora_fix, a.z_bits)
        trainable = {"full": gains, "adapters": {n: a.lora_rank for n in mats}}
        groups = [{"params": [p for A, B, _ in factors.values() for p in (A, B) if isinstance(p, torch.nn.Parameter)],
                   "lr": a.lr},
                  {"params": [params[n] for n in gains], "lr": a.norm_lr}]
        opts.append((torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, a.beta2), weight_decay=a.wd,
                                       fused=prepare.DEVICE.startswith("cuda")), None))
    else:
        # the witness checks the declared set covers every changed tensor, and a grown block's untrained tensors
        # (an adapter's zero attention) are new, so changed
        trainable = {"full": sorted(set(tuned) | set(GROWN.get("new", [])))}
        emb = ["embed"] if a.embed_lr and "embed" in mats else []
        if a.method == "muon" and mats:
            muon = [n for n in mats if n not in emb]
            opts.append((Muon([params[n] for n in muon], lr=a.lr, wd=a.wd if a.muon_wd is None else a.muon_wd), a.lr))
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
        if scales:
            return encode_ia3(theta, scales, {n: src[n] for n in gains})
        if factors:
            return prepare.encode_lora(factors, "bf16", {n: src[n] for n in gains})
        if GROWN:
            return prepare.encode_model(GROWN["config"], {**{k: theta[k] for k in GROWN["new"]},
                                                          **delta_of(src, theta, tuned)})
        return prepare.encode_full(delta_of(src, theta, tuned))

    tokens = a.batch * prepare.EVAL_CTX
    d = rec["config"]["d_model"]
    low = lowest_block(tuned)
    cuts = parse_cut(a.cut, rec["config"]["n_layer"], d, low) if a.cut else []
    crossed = [c for c in cuts if low is not None and low < c["boundary"]]
    top_delta = max((c["boundary"] for c in cuts if c["delta"]), default=None)
    grown_from = GROWN.get("start", 10 ** 6)
    refresh = frozen_flops(rec, low, min(top_delta, grown_from)) if top_delta is not None and low < grown_from else 0
    code_fwd = sum(4 * d * (c["keep"] or d) for c in cuts) + refresh
    leaky = [c for c in cuts if c["keep"]] if a.cut_leak else []
    code_train = code_fwd + sum(4 * d * (c["keep"] or d) for c in crossed) + sum(12 * d * c["keep"] for c in leaky)
    if a.cut_leak and not leaky:
        raise SystemExit("wildcard: --cut-leak penalizes what a p<keep> cut leaves out, and no cut has p<keep>")
    confine, confine_flops = {}, 0
    if a.confine:
        kept = [c for c in cuts if c["keep"]]
        confine = {n: theta[n].to(prepare.DEVICE).float() for n in tuned
                   if re.fullmatch(r"blocks\.\d+\.(wo|w_down)\.weight", n)}
        if len(kept) != 1 or not confine or factors:
            raise SystemExit("wildcard: --confine needs one p<keep> cut, full-trained wo or w_down, and no adapters")
        below = kept[0]["boundary"] - 1
        exact = ({n for n in tuned if n.startswith(f"blocks.{below}.")} == set(tuned)
                 and (below >= grown_from or tuned == [f"blocks.{below}.w_down.weight"]))
        if not exact:
            raise SystemExit(f"wildcard: --confine keeps the effect in the cut's span only if every trained tensor is "
                             f"in block {below} (the one below the cut) and that block is grown (--grow) or trains "
                             f"w_down alone: an existing block's frozen w_down carries a trained wo's effect out of it")
        span = fresh_rotation(d, 1000 + kept[0]["boundary"], prepare.DEVICE)[: kept[0]["keep"]]
        confine_flops = sum(4 * kept[0]["keep"] * d * w.shape[1] for w in confine.values())
    if cuts:
        install_cuts(net, cuts, theta, low, grown_from)
        CUT["leak"] = [] if leaky else None
        print(json.dumps({"cuts": cuts, "gradient_crosses": [c["boundary"] for c in crossed],
                          "code_flops_per_token": {"forward": code_fwd, "train": code_train,
                                                   "frozen_recompute": refresh}}), flush=True)
    cut_probe = probe_rows(rec, a)[: a.cut_probe_rows] if cuts and a.cut_probe_rows else None
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"] + extra
    per_step += code_train * tokens + confine_flops
    alphas = [1.0] + a.extrapolate if a.extrapolate else []
    reserve = probe_flops(a, rec, trainable, len(alphas)) if alphas else 0
    if cut_probe is not None:
        probe_tokens = len(cut_probe) * prepare.EVAL_CTX
        reserve += 4 * prepare.train_flops(rec, trainable, [{"op": "forward", "tokens": probe_tokens}])[
            "flops"] + 2 * code_fwd * probe_tokens
        reserve += 4 * len(cuts) * (probe_tokens * d * d + (probe_tokens // SESSION + 1) * d ** 3)
    calibrate = any(c["mean"] or c["ecode_fwd"] for c in cuts)
    if calibrate:
        calib_tokens = a.micro * prepare.EVAL_CTX
        reserve += prepare.train_flops(rec, trainable, [{"op": "forward", "tokens": calib_tokens}])["flops"]
        reserve += code_fwd * calib_tokens
    n_steps = int((flop_budget(a) - reserve) // per_step)
    if n_steps < 1:
        raise Skip(f"wildcard {a.method}: the budget leaves no step after the line search's reserve")
    base = [[g["lr"] for g in o.param_groups] for o, _ in opts]
    if scales and a.extrapolate:
        raise SystemExit("wildcard: --extrapolate moves θ's tensors or LoRA factors, not --ia3's scales")
    if a.ema and (factors or scales):
        raise SystemExit("wildcard: --ema is for full-tensor sets")
    ema = {n: params[n].detach().clone() for n in tuned} if a.ema else None
    steps, snaps = [], []
    marks = {round(n_steps * k / (a.snaps + 1)) for k in range(1, a.snaps + 1)}
    print(json.dumps({"steps": n_steps, "tuned": tuned, "flops_per_step": per_step, "extra": extra}), flush=True)
    probes = {}

    def cut_probes(when: str):
        """The probe windows' loss with every cut coded and with none (the evaluator scores the uncoded model). Only
        the end probe measures each code's spectrum, since spectrum_add reads values back and the capture marks a
        run data-dependent from a read-back before its traced step."""
        with step_of():
            for on in (True, False):
                CUT["on"] = on
                if on and when == "end":
                    CUT["spectrum"] = {}
                probes[f"{when}_{'coded' if on else 'uncoded'}"] = probe_loss(fwd, cut_probe, a.micro, steps)
                if CUT.get("spectrum") is not None:
                    probes[f"{when}_spectrum"] = spectrum_report(CUT.pop("spectrum"))
                if on:
                    steps.append({"op": "extra", "flops": code_fwd * len(cut_probe) * prepare.EVAL_CTX,
                                  "what": "cut code (probe)"})
            CUT["on"] = True
        print(json.dumps({"cut_probe": when, **probes}), flush=True)

    if calibrate:
        with step_of():
            CUT["calibrate"] = True
            probe_loss(fwd, data.next(a.micro), a.micro, steps)
            CUT["calibrate"] = False
            steps.append({"op": "extra", "flops": code_fwd * calib_tokens, "what": "cut code (calibration)"})
        print(json.dumps({"cut_mean": {b: float(m.pow(2).mean().sqrt()) for b, m in CUT_MEAN.items()}}), flush=True)
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
                if CUT.get("leak"):
                    leak = sum(CUT["leak"]) * (a.cut_leak * len(x) / a.batch)
                    CUT["leak"].clear()
                    (loss + leak).backward()
                else:
                    loss.backward()
            torch.nn.utils.clip_grad_norm_(trained, a.clip)
            for o, _ in opts:
                o.step()
                o.zero_grad(set_to_none=True)
            if confine:
                with torch.no_grad():
                    for n, w0 in confine.items():
                        params[n].copy_(w0 + span.T @ (span @ (params[n] - w0)))
            steps.append({"op": "train", "tokens": tokens})
            if confine:
                steps.append({"op": "extra", "flops": confine_flops, "what": "confine the output updates to the cut's span"})
            if extra:
                steps.append({"op": "extra", "flops": extra, "what": "muon newton-schulz"})
            if cuts:
                steps.append({"op": "extra", "flops": code_train * tokens, "what": "cut code"})
            if ema is not None:
                with torch.no_grad():
                    for n in tuned:
                        ema[n].lerp_(params[n].detach(), 1 - a.ema)
            if step + 1 in marks and not GROWN:
                snaps.append((encode(ema if ema is not None else params), len(steps)))
                if getattr(a, "probe", None) is not None:
                    print(json.dumps({"step": step + 1, "flops": prepare.train_flops(rec, trainable, steps)["flops"],
                                      "probe_loss": probe_loss(fwd, a.probe, a.micro, steps)}), flush=True)
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
    info = {"steps": n_steps, "lora_rank": a.lora_rank}
    if GROWN:
        info["grown"] = {"config": GROWN["config"], "new_from_block": GROWN["start"],
                         "snapshots": "none: te_core takes checkpoints from the encoders it wraps, and encode_model "
                                      "isn't one, so only the submission is scored"}
        if GROWN.get("adapter"):
            j, r = GROWN["start"], GROWN["adapter"]
            info["grown"]["adapter"] = {"live_units": r, "z_bits": a.adapter_z, "z_rotate": a.z_rotate,
                                        "dead_entries_nonzero": int(sum(
                int((params[f"blocks.{j}.{m}.weight"][r:] != 0).sum()) for m in ("w_gate", "w_up"))
                + int((params[f"blocks.{j}.w_down.weight"][:, r:] != 0).sum())
                + sum(int((params[f"blocks.{j}.{m}.weight"] != 0).sum()) for m in ("wq", "wk", "wv", "wo")))}
    if cuts:
        if cut_probe is not None:
            cut_probes("end")
        info["cuts"] = {"codes": cuts, "gradient_crosses": [c["boundary"] for c in crossed], "probe": probes,
                        "probe_rows": 0 if cut_probe is None else len(cut_probe)}
        if ECODE:
            info["cuts"]["ecode"] = ecode_report()
            print(json.dumps({"ecode": info["cuts"]["ecode"]}), flush=True)
        if RCODE:
            info["cuts"]["rcode"] = rcode_report()
            print(json.dumps({"rcode": info["cuts"]["rcode"]}), flush=True)
    if a.z_bits:
        info["z_code"] = {"bits": a.z_bits, "rank": a.lora_rank, "a_bits_per_direction": a.lora_rank * a.z_bits + 16}
    if scales:
        lo, hi = torch.cat([l.detach().flatten() for l in scales.values()]).aminmax()
        info["ia3"] = {"scales": {n: int(l.numel()) for n, l in scales.items()},
                       "values": int(sum(l.numel() for l in scales.values())), "scale_range": [float(lo), float(hi)]}
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
    n_steps = int((flop_budget(a) - reserve) // (per_step + moment_flops)) // rounds
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
    n_pos = int(flop_budget(a) // (fill_tok + a.epochs * per_pos))
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
    global MAX_E_BYTES
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mechanism", default="", help="the --note's mechanism name, kept in the notes")
    ap.add_argument("--set", dest="tset", default="mlp-in:-1")
    ap.add_argument("--method", default="adamw", choices=("adamw", "muon", "kfac", "cached", "scale"))
    ap.add_argument("--lora-rank", type=int, default=0,
                    help="train the set's matrices through rank-r adapters (E `encode_lora`); -1 fits --max-e-bytes")
    ap.add_argument("--max-e-bytes", type=int, default=0, help="the E size --lora-rank -1 fits")
    ap.add_argument("--lora-fix", default="", choices=("", "A", "B"),
                    help="freeze the adapters' A (random) or B (seeded orthonormal, A = 0 trained)")
    ap.add_argument("--z-bits", type=int, default=0, help="with --lora-fix B: hand z = A·h on at this many bits per coefficient")
    ap.add_argument("--ia3", action="store_true",
                    help="IA3: train a scale per output channel of each --set matrix (W' = diag(l)·W; IA3 proper is "
                         "wk, wv and w_up), E = encode_lora2's magnitudes (add_ia3)")
    ap.add_argument("--cut", default="", help="B:[d][n]q<bits>[m][e|r][p<keep>][o<k>][b<bits>[s|e]][g<group>],...: block "
                                            "boundaries handed on coded (parse_cut)")
    ap.add_argument("--grow", type=int, default=0,
                    help="train a model of θ's plus this many new top blocks (zero output matrices, so θ's function at "
                         "the start); E is the grown model (`encode_model`)")
    ap.add_argument("--grow-copy", action="store_true",
                    help="with --grow n: new block i starts as a copy of θ's block ⌊iL/n⌋ (stacking growth), its wo and "
                         "w_down still zeroed")
    ap.add_argument("--adapter", type=int, default=0,
                    help="with --grow 1: the new block is a bottleneck adapter of this many live MLP hidden units "
                         "(attention all zero; train its mlp_norm, w_gate, w_up and w_down)")
    ap.add_argument("--adapter-z", type=int, default=0,
                    help="hand an --adapter's live hidden units to its w_down as a code of this many bits each")
    ap.add_argument("--prefix", type=int, default=0,
                    help="prefix tuning: this many zero-gated learned key/value pairs in every block (`extend`); E is "
                         "the model with them (`encode_model`)")
    ap.add_argument("--block-adapter", type=int, default=0,
                    help="a bottleneck adapter of this many units after every block's MLP, its up-projection from zero "
                         "(`extend`); E is the model with them (`encode_model`)")
    ap.add_argument("--block-adapter-z", type=int, default=0,
                    help="hand every --block-adapter's units to its up-projection as a code of this many bits each")
    ap.add_argument("--z-rotate", action="store_true",
                    help="code --adapter-z's and --block-adapter-z's units in a randomized-DCT basis, as the cut codes do")
    ap.add_argument("--confine", action="store_true",
                    help="after every step, project each trained wo and w_down's update onto the p<keep> cut's span, so "
                         "the whole effect lies in what the cut hands on; everything trained must be in the block "
                         "below the cut, a grown block or that block's w_down alone")
    ap.add_argument("--cut-leak", type=float, default=0.0,
                    help="add this times the share of h - h_frozen a p<keep> cut leaves out (its mean square over "
                         "h_frozen's) to the training loss")
    ap.add_argument("--cut-probe-rows", type=int, default=128,
                    help="with --cut: probe windows scored coded and uncoded at theta and at the end (0: none)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--norm-lr", type=float, default=1e-3)
    ap.add_argument("--embed-lr", type=float, default=0.0,
                    help="the tied embedding's own AdamW rate when it trains (0: it goes with the other matrices)")
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--muon-wd", type=float, default=None,
                    help="muon: the block matrices' decoupled weight decay alone (the embedding and gains keep --wd)")
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
    a = ap.parse_args()
    MAX_E_BYTES = a.max_e_bytes
    torch.backends.cuda.matmul.allow_tf32 = True
    configs = json.loads(a.suite) if a.suite else [{}]
    if a.suite and a.lora_rank:
        raise SystemExit("wildcard: a suite trains full tensors (adapters' hooks would outlive their config)")
    net, rec, theta = prepare.load_start(a.start)
    start_theta = theta
    if a.adapter and (a.grow != 1 or a.suite or not 0 < a.adapter < net.inner.size.d_ff):
        raise SystemExit("wildcard: --adapter r is one grown block (--grow 1, no --suite) with 0 < r < d_ff live units")
    if a.adapter_z and (not a.adapter or a.adapter % 8):
        raise SystemExit("wildcard: --adapter-z codes an --adapter's live units, r a multiple of 8 (whole bytes)")
    if a.block_adapter_z and (not a.block_adapter or a.block_adapter % 8):
        raise SystemExit("wildcard: --block-adapter-z codes a --block-adapter's units, r a multiple of 8 (whole bytes)")
    if a.grow_copy and (not a.grow or a.adapter):
        raise SystemExit("wildcard: --grow-copy starts --grow's new blocks as copies of θ's (not an --adapter)")
    if a.z_rotate and not (a.adapter_z or a.block_adapter_z):
        raise SystemExit("wildcard: --z-rotate rotates an --adapter-z or --block-adapter-z code")
    ext = {k: n for k, n in (("prefix", a.prefix), ("adapter", a.block_adapter)) if n}
    if len(ext) > 1 or ext and (a.grow or a.lora_rank or a.ia3 or a.suite):
        raise SystemExit("wildcard: --prefix and --block-adapter are each their own update (no --grow, --lora-rank, "
                         "--ia3, --suite or the other)")
    if a.grow:
        rec, theta = grow(net, rec, start_theta, a.grow, 7919 + a.data_seed, a.adapter, a.grow_copy)
    for key, n in ext.items():
        rec, theta = extend(net, rec, start_theta, key, n, 7919 + a.data_seed)
    params = net.params()
    tuned = resolve_set(a.tset, rec, params)
    shape = prepare.deep_share(rec, set(tuned))
    print(json.dumps({"set": tuned, "lowest_updated_layer": shape["lowest_updated_layer"]}), flush=True)
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
    model = prepare.decode(start_theta, blob)
    changed = sorted({n for n in tuned if n not in start_theta or not prepare.same_bits(model[n], start_theta[n])}
                     | set(GROWN.get("new", [])))
    notes = {"direction": "wildcard", "mechanism": a.mechanism, "method": c.method, "set": tuned,
             "args": {k: v for k, v in vars(c).items() if k != "probe"}, "info": info, "E_bytes": len(blob),
             "suite": results if a.suite else None, "source": Path(__file__).read_text()}
    prepare.submit(Path(a.out), a.start, blob, model, trainable, all_steps, changed=changed, notes=notes,
                   snapshots=snaps if not a.suite else [])
    return 0

def probe_rows(rec: dict, a) -> torch.Tensor:
    """The last `--probe-rows` windows of the training order, which a run reaches only after reading all the others."""
    d = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    d.at = len(d.order) - a.probe_rows
    return d.next(a.probe_rows)


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
