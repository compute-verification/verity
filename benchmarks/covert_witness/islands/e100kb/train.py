"""Islands `e1mb` and `e100kb` (direction `lowrank`): a low-rank update to the blocks' matrices, sized to the island's
own size (E_BYTES: 1,000,000 bytes in e1mb, 100,000 in e100kb; nothing caps it) or to `--e-bytes` (for the A2 curve over
Delta); E is the factors and any norm gains, and the evaluated model merges them. The same file runs in both islands.

The base stays frozen and the forward adds s · B(A x) to each target's output (a hook, so the base matmuls read θ and only
the adapters' MACs are updated). Knobs, each a mechanism of the direction:

- placement: `--targets` (all, out, attn, mlp, qkv, qv, none, or leaf names), `--layers` (all, top:N, A-B, or a list);
- size: `--rank` (0: the largest uniform rank that fits `--e-bytes`, default E_BYTES less 512, or 2048 for quant),
  `--rank-map leaf=R,...`, or `--alloc grad`: each target's rank by its full gradient over `--init-batches` batches, rank
  units taken in decreasing σ² per byte of E (σ the gradient's singular values) until E fills `--e-bytes` (|E| from an
  affine model of the ranks fitted to exact encodes before the probe: an E encoded after compute is a checkpoint);
- norm gains: `--norms` (none, all, final, adapted: the adapted blocks' and the final norm);
- factor precision, `--format`:
  - lora (CWLORA1, `prepare.encode_lora`): `--dtype` bf16 or f16 factors (f16 rescales A and B to unit max and moves the
    factor into the scale);
  - lora2 (CWLORA2, `prepare.encode_lora2`; `--bits` 4 or 8): integer factors, one f16 scale per rank component on each
    side, the gains in `--dtype`; `--dora` adds DoRA magnitudes, one per output row in f16 (`--dtype f16`), each row of
    W + sBA rescaled to its magnitude;
  - quant (CWQUAN1, `prepare.encode_quant`; `--bits` 1 to 8, `--bits-map leaf=B` or `leaf=BA:BB` and `--b-bits` for
    mixed precision): each pair balanced (B's columns to unit RMS, those RMS moved into A's rows), A by row and B as one
    group coded into the Lloyd–Max codebook of N(0, 1) times an f16 RMS scale; the gains as `--norm-bits` deltas from θ
    (uniform codes at the absmax, 0: in full), xz'd;

  quantized factors and magnitudes train through their rounding (straight-through, the rounding refreshed once a step);
- initialization: `--init` gauss (A ~ N(0, 1/d_in), B = 0), orth (A orthonormal rows), grad (A the top-r right singular
  vectors of the targets' full gradient over `--init-batches` batches, B = 0; the pass and the weight gradients are logged);
- optimizer: AdamW with `--b-lr-mult` (LoRA+), `--opt riemann` (each factor's gradient preconditioned by the other's Gram,
  Zhang and Pilanci 2024), `--opt muon` (A and B by Muon at `--lr`: Nesterov momentum, one fp32 buffer per value,
  orthogonalized by 5 Newton–Schulz steps, its matmuls counted as extra FLOPs; gains and magnitudes stay on AdamW),
  `--beta2`, `--wd`;
- schedule: `--schedule` linear (warmup, linear decay to 0), wsd (`--cooldown` fraction), cosine; `--batch`;
- data: `--fresh` skips the windows the start trained on (its seed-0 order's prefix).

The run plans its steps for the harness (`hooks.plan`) and encodes E in the middle of each of `--snaps` + 1 shares of them
(the default 19: each 5%), so each of the harness's 5% checkpoints scores one. The record's notes carry `delta_grad`: the
trainable values and the bits of their gradient, of the optimizer's state (AdamW's m and v, or Muon's buffer, fp32) and of
the fp32 masters the update rule reads and writes beside what E stores (the factors, any magnitudes and the norm gains:
`latent_bits`, Δ_t).

    python launch.py START --island e1mb --node NODE --note TEXT -- [--targets all] [--rank 0] ...
"""
from __future__ import annotations

import argparse
import functools
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
#: the island's update size, its own choice (nothing caps it); `--e-bytes` sets another
E_BYTES = {"e1mb": 1_000_000, "e100kb": 100_000}.get(KLASS, 1_000_000)
LF_MATS = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")
HF_OF = {"wq": "q_proj", "wk": "k_proj", "wv": "v_proj", "wo": "o_proj", "w_gate": "gate_proj", "w_up": "up_proj",
         "w_down": "down_proj"}
GROUPS = {"all": set(LF_MATS), "out": {"wo", "w_down"}, "attn": {"wq", "wk", "wv", "wo"}, "mlp": {"w_gate", "w_up", "w_down"},
          "qkv": {"wq", "wk", "wv"}, "qv": {"wq", "wv"}, "none": set()}
FORMATS = ("lora", "lora2", "quant")


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


@functools.lru_cache(maxsize=None)
def codebook(bits: int, uniform: bool = False) -> torch.Tensor:
    """The f16 codebook CWQUAN1 stores, increasing: the 2^bits-level Lloyd–Max quantizer of N(0, 1), or (uniform)
    (k − h) / (h − 1) for k < 2h, h = 2^(bits − 1)."""
    k = 2 ** bits
    if uniform:
        return ((torch.arange(k, dtype=torch.float64) - k // 2) / (k // 2 - 1)).to(torch.float16)
    xs = torch.linspace(-8, 8, 40001, dtype=torch.float64)
    w = torch.exp(-xs ** 2 / 2)
    c = torch.special.ndtri((torch.arange(k, dtype=torch.float64) + 0.5) / k)
    for _ in range(200):
        idx = torch.bucketize(xs, (c[1:] + c[:-1]) / 2)
        num = torch.zeros(k, dtype=torch.float64).index_add_(0, idx, xs * w)
        c = num / torch.zeros(k, dtype=torch.float64).index_add_(0, idx, w)
    return c.to(torch.float16)


@functools.lru_cache(maxsize=None)
def codebook_on(bits: int, uniform: bool, device: str) -> torch.Tensor:
    return codebook(bits, uniform).float().to(device)


def qcodes(x: torch.Tensor, bits: int, group: int, uniform: bool = False) -> tuple[prepare.QArray, torch.Tensor]:
    """x's entries in row-major groups of `group` as `bits`-bit codes into `codebook` times an f16 scale per group (the
    group's RMS, or its max |x| when uniform): the QArray and the fp32 values E rebuilds from it, in x's shape."""
    cb = codebook_on(bits, uniform, str(x.device))
    g = x.detach().float().reshape(-1, group)
    sc = (g.abs().amax(1) if uniform else g.pow(2).mean(1).sqrt()).to(torch.float16)
    scf = sc.float()[:, None]
    codes = torch.bucketize(g / torch.where(scf > 0, scf, 1.0), (cb[1:] + cb[:-1]) / 2)
    q = prepare.QArray(codebook(bits, uniform), sc, codes.to(torch.uint8), bits, group, tuple(x.shape))
    return q, (cb[codes] * scf).view(x.shape)


def quant_pair(A: torch.Tensor, B: torch.Tensor, bits_a: int, bits_b: int):
    """A (r x d_in) and B (d_out x r) as CWQUAN1 codes: B's columns over their RMS c, A's rows times c over G (the largest
    such row's RMS), A coded by row and B as one group. Returns (Q_A, Q_B, G, A_eff, B_eff): s · B_eff A_eff is
    s · G · Q_B Q_A, the product E rebuilds at scale s · G."""
    A, B = A.detach().float(), B.detach().float()
    c = B.pow(2).mean(0).sqrt()
    c = torch.where(c > 0, c, torch.ones_like(c))
    ac = A * c[:, None]
    G = float(ac.pow(2).mean(1).sqrt().max()) or 1.0
    qa, va = qcodes(ac / G, bits_a, A.shape[1])
    qb, vb = qcodes(B / c[None, :], bits_b, B.numel())
    return qa, qb, G, va * (G / c)[:, None], vb * c[None, :]


class LowRank:
    """One target's factors (fp32 latents A, B), its scale and precision, the straight-through offsets to the values E
    stores (`refresh`), and with DoRA the magnitudes (init: W's row norms)."""

    def __init__(self, w: torch.nn.Parameter, a0: torch.Tensor, s: float, bits: tuple[int, int], fmt: str, dora: bool):
        self.w, self.s, self.bits, self.fmt = w, s, bits, fmt
        self.A = torch.nn.Parameter(a0.float().clone())
        self.B = torch.nn.Parameter(torch.zeros(w.shape[0], a0.shape[0], device=a0.device))
        quantized = fmt != "lora"
        self.dA = torch.zeros_like(self.A) if quantized else None
        self.dB = torch.zeros_like(self.B) if quantized else None
        self.wn2 = w.detach().float().pow(2).sum(1) if dora else None
        self.m = torch.nn.Parameter(self.wn2.sqrt().clone()) if dora else None
        self.dm = torch.zeros_like(self.m) if dora else None

    def latents(self) -> int:
        """The fp32 values the update rule reads and writes beside what E stores: the factors (bf16 or coded in E) and
        any magnitudes."""
        return self.A.numel() + self.B.numel() + (self.m.numel() if self.m is not None else 0)

    @torch.no_grad()
    def refresh(self, mag_dtype: torch.dtype) -> None:
        if self.fmt == "lora2":
            fa, fb = prepare.lora2_factors(self.A, self.B, self.bits[0])
        elif self.fmt == "quant":
            *_, fa, fb = quant_pair(self.A, self.B, *self.bits)
        if self.dA is not None:
            self.dA.copy_(fa - self.A)
            self.dB.copy_(fb - self.B)
        if self.m is not None:
            self.dm.copy_(self.m.to(mag_dtype).float() - self.m)

    def hook(self, mod, inp, out):
        fa = self.A if self.dA is None else self.A + self.dA
        fb = self.B if self.dB is None else self.B + self.dB
        y = out + self.s * F.linear(F.linear(inp[0], fa), fb)
        if self.m is None:
            return y
        # each output row of W + sBA rescaled to its magnitude, the row norms and the rescale in fp32
        with torch.autocast(prepare.DEVICE.split(":")[0], enabled=False):
            fa32, fb32 = fa.float(), fb.float()
            cross = (fb32 * (self.w.float() @ fa32.t())).sum(1)
            sq = ((fb32 @ (fa32 @ fa32.t())) * fb32).sum(1)
            n = (self.wn2 + 2 * self.s * cross + self.s ** 2 * sq).clamp_min(1e-12).sqrt()
            return (y.float() * ((self.m + self.dm) / n)).to(y.dtype)


def bits_of(targets: list[str], a) -> dict[str, tuple[int, int]]:
    """Each target's (A's bits, B's bits): `--bits`, then `--b-bits` for B, then `--bits-map leaf=B` or `leaf=BA:BB`."""
    out = {n: (a.bits, a.b_bits or a.bits) for n in targets}
    for item in (x for x in a.bits_map.split(",") if x):
        leaf, v = item.split("=")
        ba, bb = (int(x) for x in v.split(":")) if ":" in v else (int(v), int(v))
        out.update({n: (ba, bb) for n in targets if leaf_of(n) == leaf})
    allowed = {"lora": {0}, "lora2": set(prepare.LORA2_BITS), "quant": set(range(1, 9))}[a.format]
    for n, (ba, bb) in out.items():
        if {ba, bb} - allowed or (a.format == "lora2" and ba != bb):
            raise SystemExit(f"{KLASS}: {n}: bits {ba}:{bb} for --format {a.format} (lora: 0; lora2: 4 or 8 on both "
                             "factors; quant: 1 to 8)")
    return out


def e_bytes(theta, spec: dict[str, tuple[int, int, int]], norms, a) -> int:
    """|E| for ranks and bits `spec` (name -> (r, A's bits, B's bits)) and the gains `norms`. For quant, with uniformly
    random codes and scales, which xz can't shrink, so the trained codes' E is at most about this."""
    if a.format == "lora":
        fac = {n: (torch.zeros(r, theta[n].shape[1]), torch.zeros(theta[n].shape[0], r), 1.0) for n, (r, _, _) in spec.items()}
        return len(prepare.encode_lora(fac, a.dtype, {n: theta[n] for n in norms}))
    if a.format == "lora2":
        fac = {n: (torch.zeros(r, theta[n].shape[1]), torch.zeros(theta[n].shape[0], r), 1.0) for n, (r, _, _) in spec.items()}
        mags = {n: torch.ones(theta[n].shape[0]) for n in spec} if a.dora else None
        return len(prepare.encode_lora2(fac, {n: ba for n, (_, ba, _) in spec.items()}, a.dtype, magnitudes=mags,
                                        full={n: theta[n] for n in norms}))
    g = torch.Generator().manual_seed(0)

    def rand(shape: tuple[int, ...], bits: int, group: int, uniform: bool = False) -> prepare.QArray:
        n = math.prod(shape)
        return prepare.QArray(codebook(bits, uniform), (torch.rand(n // group, generator=g) + 0.5).half(),
                              torch.randint(0, 2 ** bits, (n,), generator=g, dtype=torch.uint8), bits, group, shape)

    entries = {}
    for n, (r, ba, bb) in spec.items():
        d_out, d_in = theta[n].shape
        # a scale whose JSON is as long as any float's
        entries[n] = {"kind": "lora", "a": rand((r, d_in), ba, d_in), "b": rand((d_out, r), bb, d_out * r),
                      "scale": -1.2345678901234567e-300}
    for n in norms:
        shape = tuple(theta[n].shape)
        entries[n] = ({"kind": "delta", "q": rand(shape, a.norm_bits, math.prod(shape), True)} if a.norm_bits
                      else {"kind": "full", "t": theta[n]})
    return len(prepare.encode_quant(entries, theta))


def unit_bytes(theta, n: str, bits: tuple[int, int], a) -> float:
    """The bytes one more rank adds to target n's factors."""
    d_out, d_in = theta[n].shape
    if a.format == "lora":
        return 2 * (d_in + d_out)
    if a.format == "lora2":
        return bits[0] * (d_in + d_out) / 8 + 4
    return (bits[0] * d_in + bits[1] * d_out) / 8 + 2


def size_model(theta, targets: list[str], bits: dict, norms: list[str], a, calib: dict[str, int]):
    """|E| as an affine function of the ranks, from exact encodes made before any compute (the harness takes an E encoded
    after compute as a checkpoint): the gains alone, each target's fixed bytes at rank 1, its bytes per rank, and one
    factor matching the model to the exact |E| at the ranks `calib`."""
    spec = lambda ranks: {n: (r, *bits[n]) for n, r in ranks.items()}
    base = e_bytes(theta, {}, norms, a)
    fixed = {n: e_bytes(theta, spec({n: 1}), norms, a) - base - unit_bytes(theta, n, bits[n], a) for n in targets}
    model = lambda ranks: base + sum(fixed[n] + unit_bytes(theta, n, bits[n], a) * r for n, r in ranks.items() if r > 0)
    k = e_bytes(theta, spec(calib), norms, a) / model(calib)
    return lambda ranks: math.ceil(k * model(ranks))


def fit_prefix(size, hi: int, target: int) -> int:
    """The largest p in [0, hi] with size(p) <= target, size increasing in p."""
    if size(hi) <= target:
        return hi
    lo = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if size(mid) <= target else (lo, mid)
    return lo


def ranks_for(theta, targets: list[str], rank: int, rank_map: dict[str, int]) -> dict[str, int]:
    out = {n: min(rank_map.get(leaf_of(n), rank), *theta[n].shape) for n in targets}
    return {n: r for n, r in out.items() if r > 0}


def delta_grad(adam: list[torch.Tensor], muon: list[torch.Tensor], latent: int) -> dict:
    """Δ_grad: each trained value's fp32 gradient and its optimizer's fp32 state (AdamW's m and v, Muon's one buffer),
    plus the `latent` fp32 masters the update rule reads and writes beside what E stores."""
    na, nm = sum(p.numel() for p in adam), sum(p.numel() for p in muon)
    n = na + nm
    bits = 32 * n + 64 * na + 32 * nm + 32 * latent
    return {"trainable": n, "grad_bits": 32 * n, "adam_bits": 64 * na, "muon_bits": 32 * nm, "latent_bits": 32 * latent,
            "bits": bits, "bytes": bits // 8}


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
    the corpus's windows that its provenance's token count covers (starts.py's data order; the GPT-2 stream only)."""
    prov = rec.get("provenance", {})
    seen_windows = int(prov.get("tokens", 0)) // data.ctx
    if not seen_windows or prov.get("data_seed", 0) != 0 or rec["tokenizer"] != "gpt2":
        return
    n = int(data.starts[-1])
    seen = np.zeros(n, dtype=bool)
    seen[np.random.default_rng(0).permutation(n)[:seen_windows]] = True
    data.order = data.order[~seen[data.order]]


@torch.no_grad()
def grad_probe(net, params, names, data, a, steps, tokens) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Each target's full gradient over `init_batches` batches (eager, before the adapters exist), as its singular values
    and right singular vectors."""
    for n in names:
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
            steps.append({"op": "extra", "flops": sum(2 * params[n].numel() * tokens for n in names)})
    out = {}
    svd = 0
    for n in names:
        g = params[n].grad.float()
        _, sv, vh = torch.linalg.svd(g, full_matrices=False)
        out[n] = (sv, vh)
        svd += 4 * g.shape[0] * g.shape[1] * min(g.shape)
        params[n].requires_grad_(False)
    for p in params.values():
        p.grad = None
    steps.append({"op": "extra", "flops": svd})
    return out


@torch.no_grad()
def riemann_precondition(ads: dict, delta: float) -> int:
    """grad_A ← (BᵀB + δI)⁻¹ grad_A, grad_B ← grad_B (AAᵀ + δI)⁻¹; returns the matmul FLOPs spent."""
    flops = 0
    for ad in ads.values():
        A, B = ad.A, ad.B
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
    def __init__(self, params, lr: float, momentum: float = 0.95):
        super().__init__(params, {"lr": lr, "momentum": momentum})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state.setdefault(p, {"buf": torch.zeros_like(p)})
                buf = st["buf"].mul_(g["momentum"]).add_(p.grad)
                u = newton_schulz(p.grad.add(buf, alpha=g["momentum"]))
                p.add_(u, alpha=-g["lr"] * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default="all")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--rank", type=int, default=0, help="0: the largest uniform rank that fits --e-bytes")
    ap.add_argument("--rank-map", default="", help="leaf=R,... overriding --rank per leaf (0 drops the leaf)")
    ap.add_argument("--alloc", default="uniform", choices=("uniform", "grad"))
    ap.add_argument("--e-bytes", type=int, default=0, help="the byte target the ranks fit (default: E_BYTES less 512, or "
                                                             "2048 for quant)")
    ap.add_argument("--norms", default="all")
    ap.add_argument("--format", default="", choices=("", *FORMATS),
                    help="default: lora for --bits 0, lora2 for 4 or 8, else quant")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "f16"))
    ap.add_argument("--bits", type=int, default=0, choices=range(9))
    ap.add_argument("--b-bits", type=int, default=0, help="B's bits when not --bits (quant)")
    ap.add_argument("--bits-map", default="", help="leaf=B or leaf=BA:BB,... overriding the bits per leaf")
    ap.add_argument("--norm-bits", type=int, default=6, choices=(0, *range(2, 9)),
                    help="quant: the gains as deltas at this many bits (0: in full)")
    ap.add_argument("--dora", action="store_true", help="DoRA magnitudes (lora2, --dtype f16)")
    ap.add_argument("--mag-lr", type=float, default=3e-4)
    ap.add_argument("--init", default="gauss", choices=("gauss", "orth", "grad"))
    ap.add_argument("--init-batches", type=int, default=4)
    ap.add_argument("--scale", default="rs", choices=("rs", "one"), help="rs: s = sqrt(d_in / r); one: s = 1")
    ap.add_argument("--opt", default="adamw", choices=("adamw", "riemann", "muon"))
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
    a.format = a.format or ("lora" if not a.bits else "lora2" if a.bits in prepare.LORA2_BITS else "quant")
    if a.dora and (a.format != "lora2" or a.dtype != "f16"):
        raise SystemExit(f"{KLASS}: --dora needs --format lora2 (CWLORA2 holds the magnitudes) and --dtype f16")
    if a.opt == "muon" and a.wd:
        raise SystemExit(f"{KLASS}: --opt muon takes no --wd")
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    want = a.e_bytes or E_BYTES
    target_bytes = a.e_bytes or (E_BYTES - (2048 if a.format == "quant" else 512))
    layers = layers_of(a.layers, rec["config"]["n_layer"])
    targets = targets_of(params, a.targets, layers)
    norms = norms_of(params, a.norms, layers)
    bits = bits_of(targets, a)
    rank_map = {k: int(v) for k, v in (x.split("=") for x in a.rank_map.split(",") if x)}
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    if a.fresh:
        fresh_order(data, rec)
    tokens = a.batch * prepare.EVAL_CTX
    steps, snaps = [], []
    for p in params.values():
        p.requires_grad_(False)
    # every exact |E| comes before the first compute: an E encoded after it is a checkpoint
    size_of = lambda ranks: e_bytes(theta, {n: (r, *bits[n]) for n, r in ranks.items()}, norms, a)
    rank = a.rank
    if not rank and targets:
        hi = max(min(theta[n].shape) for n in targets)
        rank = max(1, fit_prefix(lambda r: size_of(ranks_for(theta, targets, r, rank_map)), hi, target_bytes))
    ranks = ranks_for(theta, targets, rank, rank_map)
    size = size_of(ranks)
    model = size_model(theta, targets, bits, norms, a, ranks) if a.alloc == "grad" else None
    probe = grad_probe(net, params, targets, data, a, steps, tokens) if a.init == "grad" or a.alloc == "grad" else {}
    if a.alloc == "grad":
        # rank units in decreasing σ² per byte; σ falls with the unit's index, so each target's units come in order
        units = sorted(((float(sv[j]) ** 2 / unit_bytes(theta, n, bits[n], a), n) for n, (sv, _) in probe.items()
                        for j in range(len(sv))), key=lambda u: -u[0])
        order = [n for _, n in units]

        def ranks_at(p: int) -> dict[str, int]:
            out = {}
            for n in order[:p]:
                out[n] = out.get(n, 0) + 1
            return out

        ranks = ranks_at(fit_prefix(lambda p: model(ranks_at(p)), len(order), target_bytes))
        rank, size = 0, model(ranks)
    if size > want:
        raise SystemExit(f"{KLASS}: ranks {sorted(set(ranks.values()))} on {len(ranks)} weights and {len(norms)} norms "
                         f"is {size} bytes, over {want}")
    for n in norms:
        params[n].data = params[n].data.float()
        params[n].requires_grad_(True)
    g = torch.Generator(device=prepare.DEVICE).manual_seed(a.seed)
    ads = {}
    for n, r in ranks.items():
        w = params[n]
        if a.init == "grad":
            a0 = probe[n][1][:r].contiguous()
        elif a.init == "orth":
            q, _ = torch.linalg.qr(torch.randn(w.shape[1], r, device=prepare.DEVICE, generator=g))
            a0 = q.T.contiguous()
        else:
            a0 = torch.randn(r, w.shape[1], device=prepare.DEVICE, generator=g) / math.sqrt(w.shape[1])
        s = math.sqrt(w.shape[1] / r) if a.scale == "rs" else 1.0
        ads[n] = LowRank(w, a0, s, bits[n], a.format, a.dora)
        net.inner.get_submodule(n.removesuffix(".weight")).register_forward_hook(ads[n].hook)
    del probe
    mag_dtype = prepare.DTYPES[a.dtype]
    for ad in ads.values():
        ad.refresh(mag_dtype)
    fwd = torch.compile(net)
    factors = [{"params": [ad.A for ad in ads.values()], "lr": a.lr, "weight_decay": a.wd},
               {"params": [ad.B for ad in ads.values()], "lr": a.lr * a.b_lr_mult, "weight_decay": a.wd}]
    rest = [{"params": [ad.m for ad in ads.values() if ad.m is not None], "lr": a.mag_lr, "weight_decay": 0.0},
            {"params": [params[n] for n in norms], "lr": a.norm_lr, "weight_decay": 0.0}]
    muon = [grp for grp in factors if grp["params"]] if a.opt == "muon" else []
    adam = [grp for grp in (rest if a.opt == "muon" else factors + rest) if grp["params"]]
    opts = ([Muon(muon, lr=a.lr)] if muon else []) + (
        [torch.optim.AdamW(adam, betas=(0.9, a.beta2), fused=prepare.DEVICE.startswith("cuda"))] if adam else [])
    extra = sum(ns_flops(tuple(p.shape)) for grp in muon for p in grp["params"])
    trainable = {"full": norms, "adapters": ranks}
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"] + extra
    spent = prepare.train_flops(rec, trainable, steps)["flops"]
    n_steps = int((a.budget_fraction * prepare.budget(a.start)["train_flops"] - spent) // per_step)
    # mid-share, so a rounded step never puts two encodes in one harness window and none in the next
    marks = {round(n_steps * (k + 0.5) / (a.snaps + 1)) for k in range(a.snaps + 1)}
    hooks.plan(n_steps)
    trained = [p for grp in muon + adam for p in grp["params"]]
    dgrad = delta_grad([p for grp in adam for p in grp["params"]], [p for grp in muon for p in grp["params"]],
                       sum(ad.latents() for ad in ads.values()) + sum(params[n].numel() for n in norms))
    print(json.dumps({"island": KLASS, "format": a.format, "opt": a.opt, "steps": n_steps, "rank": rank,
                      "ranks": sorted(set(ranks.values())), "rank_sum": sum(ranks.values()), "targets": len(ranks),
                      "norms": len(norms), "E_bytes": size, "delta_grad": dgrad, "flops_per_step": per_step,
                      "extra_per_step": extra, "init_flops": spent, "windows": len(data.order)}), flush=True)
    t0 = time.perf_counter()
    groups = [grp for opt in opts for grp in opt.param_groups]
    lrs = [grp["lr"] for grp in groups]
    ema = None
    for step in range(n_steps):
        frac = lr_factor(step, n_steps, a)
        for grp, lr in zip(groups, lrs):
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
            steps.append({"op": "extra", "flops": riemann_precondition(ads, a.riemann_delta)})
        if extra:
            steps.append({"op": "extra", "flops": extra, "what": "muon newton-schulz"})
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        for ad in ads.values():
            ad.refresh(mag_dtype)
        ema = total if ema is None else 0.98 * ema + 0.02 * total
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": round(total, 4), "ema": round(ema, 4),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if step + 1 in marks:
            snaps.append((encode(ads, params, norms, theta, a), len(steps)))
    blob = encode(ads, params, norms, theta, a)
    model = prepare.decode(theta, blob)
    prepare.submit(Path(a.out), a.start, blob, model, trainable, steps, changed=list(ranks) + norms,
                   notes={"method": "lora", "direction": "lowrank", "format": a.format, "rank": rank, "ranks": ranks,
                          "bits": {n: list(bits[n]) for n in ranks}, "args": vars(a), "steps": n_steps,
                          "final_train_ema": ema, "E_bytes": len(blob), "delta_grad": dgrad}, snapshots=snaps)
    return 0


def encode(ads: dict, params: dict, norms: list[str], theta: dict, a) -> bytes:
    full = {n: params[n] for n in norms}
    if a.format == "lora2":
        return prepare.encode_lora2({n: (ad.A.detach().float(), ad.B.detach().float(), ad.s) for n, ad in ads.items()},
                                    {n: ad.bits[0] for n, ad in ads.items()}, a.dtype,
                                    magnitudes={n: ad.m.detach() for n, ad in ads.items() if ad.m is not None}, full=full)
    if a.format == "quant":
        entries = {}
        for n, ad in ads.items():
            qa, qb, G, _, _ = quant_pair(ad.A, ad.B, *ad.bits)
            entries[n] = {"kind": "lora", "a": qa, "b": qb, "scale": ad.s * G}
        for n in norms:
            d = params[n].detach().float() - theta[n].to(params[n].device).float()
            entries[n] = ({"kind": "delta", "q": qcodes(d, a.norm_bits, d.numel(), uniform=True)[0]} if a.norm_bits
                          else {"kind": "full", "t": params[n]})
        return prepare.encode_quant(entries, theta)
    out = {}
    for n, ad in ads.items():
        A, B, s = ad.A.detach().float(), ad.B.detach().float(), ad.s
        if a.dtype == "f16":
            ma, mb = float(A.abs().max()), float(B.abs().max())
            if ma > 0 and mb > 0:
                A, B, s = A / ma, B / mb, s * ma * mb
        out[n] = (A, B, s)
    return prepare.encode_lora(out, a.dtype, full)


if __name__ == "__main__":
    raise SystemExit(main())
