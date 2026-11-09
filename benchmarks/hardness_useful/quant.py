"""Per-output-channel int8 on every block linear, with the scale chosen so Strassen's sums overflow int8 at a target rate.

A row's scale is s = min(c·σ_row, absmax_row) / 127 and its codes q = clamp(round(w / s), −127, 127): c = None is
the stock absmax scale, and a smaller c clips the row's tail and spreads its codes over more of int8. c is one number
per matrix, solved by bisection so that the matrix's Strassen level-1 overflow (`overflow`, HT-PREMISE's
`natural.overflow_profile`: the fraction of entries of the seven level-1 W-forms of C = qᵀ outside [−128, 127]) is at
least the target; a matrix already at or above it keeps absmax.

`QLinear` is the straight-through fake-quantized linear: the forward computes (x·qᵀ)·s with x in bf16, q exact in
bf16, fp32 accumulation and the scale applied in fp32; the backward passes the gradient to the latent fp32 weight inside
the clip and zero outside. The scale is recomputed from the
latent weight on every training forward (no gradient), and frozen with the codes in eval mode.

With rotate, each linear's latent weight is W·R and its input x·R (`rotation.py`: a random-sign Hadamard rotation, exact
in real arithmetic), so the codes quantize a near-Gaussian row and the clip a target needs spares the outliers.

With `add_lora`, a QLinear's latent weight is W + B·A, W frozen and A, B (rank r) trained, and every code, scale and c
is of that sum, so what is served is one matrix of codes, as without it; `merge_lora` folds B·A into W.

`Served` is how the activations reach the codes: bf16, int8 per token, or design U's int8 plus live 2-bit noise, each
matrix in its own mode (`set_served`).
"""

from __future__ import annotations

import ctypes
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import rotation

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# mode: (the E it reads, whether its activations are rounded to int8, E's mean folded out per output)
MODES = {"a16": (None, False, 0.0), "a8": (None, True, 0.0), "e2": ("e4", True, 0.5), "e2f": ("e4", False, 0.5),
         "e1": ("e4", True, 0.5), "e3": ("e3", True, 0.0)}
S_V = [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
W_V = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 1, -1], [-1, 1, 0, 0], [1, -1, 0, 1], [0, -1, 0, 1]]
SCHEMES = {"strassen": S_V, "winograd": W_V}
CALIB_TOL = 1e-6
FIELDS = torch.arange(0, 62, 2)


def draw(kind: str, shape, g: torch.Generator, device) -> torch.Tensor:
    """One call's E as int8: e4 uniform on {−2, −1, 0, 1}, e3 on {−1, 0, 1}. On a GPU one randint an element; on the
    CPU, whose generator draws one value at a time (otherwise most of a 70B forward), 2-bit fields cut from 62-bit
    draws, e3's by rejecting 3."""
    lo = -2 if kind == "e4" else -1
    if device.type != "cpu":
        return torch.randint(lo, 2, shape, device=device, generator=g).to(torch.int8)
    n, out, got = shape.numel(), [], 0
    while got < n:
        m = n - got if kind == "e4" else (n - got) * 4 // 3 + 4096
        w = torch.empty((m + 30) // 31, dtype=torch.int64).random_(0, 2**62, generator=g)
        f = (w[:, None] >> FIELDS).to(torch.int8).bitwise_and_(3).reshape(-1)[:m]
        out.append(f if kind == "e4" else f[f != 3])
        got += out[-1].numel()
    return (torch.cat(out) if len(out) > 1 else out[0])[:n].reshape(shape) + lo


def row_scale(w: torch.Tensor, c: float | None) -> torch.Tensor:
    w = w.detach().float()
    amax = w.abs().amax(1)
    s = amax if c is None else torch.minimum(c * w.pow(2).mean(1).sqrt(), amax)
    return (s / 127).clamp_min(1e-12)


def codes(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return torch.round((w.detach().float() / s[:, None]).clamp(-127, 127))


def overflow(q: torch.Tensor, depth: int = 1) -> dict[str, list[float]]:
    """The fraction of each level's W-form entries outside int8, per scheme, as `natural.overflow_profile`."""
    C = q.to(torch.int16).T
    out = {}
    for sname, V in SCHEMES.items():
        level, frac = [C], []
        for _ in range(depth):
            nxt, over, tot = [], 0, 0
            for X in level:
                h, w = X.shape[0] // 2, X.shape[1] // 2
                Bb = [X[:h, :w], X[:h, w:], X[h:, :w], X[h:, w:]]
                for v in V:
                    Cv = sum(c * Bb[i] for i, c in enumerate(v) if c)
                    over += int(((Cv < -128) | (Cv > 127)).sum())
                    tot += Cv.numel()
                    nxt.append(Cv)
            frac.append(over / tot)
            level = nxt
        out[sname] = frac
    return out


def strassen1(w: torch.Tensor, c: float | None) -> float:
    return overflow(codes(w, row_scale(w, c)), 1)["strassen"][0]


def calibrate(w: torch.Tensor, target: float | None, iters: int = 30) -> tuple[float | None, float]:
    """(c, its Strassen level-1 overflow): the largest c whose overflow is at least target; None (absmax) if absmax is."""
    base = strassen1(w, None)
    if target is None or base >= target:
        return None, base
    wf = w.detach().float()
    hi = float((wf.abs().amax(1) / wf.pow(2).mean(1).sqrt().clamp_min(1e-30)).max())  # c at which every row is absmax
    lo = 0.25
    assert strassen1(w, lo) >= target, "even c = 0.25 is under the target"
    for _ in range(iters):
        mid = (lo + hi) / 2
        if strassen1(w, mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo, strassen1(w, lo)


class _FQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, s):
        u = w / s[:, None]
        ctx.save_for_backward(u.abs() <= 127, s)
        return torch.round(u.clamp(-127, 127)).to(torch.bfloat16)

    @staticmethod
    def backward(ctx, g):
        mask, s = ctx.saved_tensors
        return g.float() * mask / s[:, None], None


class _FQLoRA(torch.autograd.Function):
    """`_FQ` of the latent W + B·A at clip c, W frozen: (codes, scales), the scales of the sum and without gradient, and
    the gradient to A and B only. The clip's mask is recomputed from the sum in the backward rather than saved (an n×K
    bool per matrix), and the sum is formed in fp32 whatever the autocast."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, w, a, b, c):
        u = torch.addmm(w, b, a)
        s = row_scale(u, c)
        ctx.save_for_backward(w, a, b, s)
        ctx.mark_non_differentiable(s)
        return torch.round(u.div_(s[:, None]).clamp_(-127, 127)).to(torch.bfloat16), s

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, g, _gs):
        w, a, b, s = ctx.saved_tensors
        u = torch.addmm(w, b, a).div_(s[:, None])
        gu = g.float().mul_(u.abs_() <= 127).div_(s[:, None])
        return None, b.T @ gu, gu @ a.T, None


class Rotation(nn.Module):
    """x ↦ x·R over the last dimension (`rotation.py`), in the input's dtype."""

    def __init__(self, k: int):
        super().__init__()
        self.m, self.a, self.b = rotation.factors(k)
        self.register_buffer("d", torch.from_numpy(rotation.signs(k)).float())
        self.register_buffer("ha", torch.from_numpy(rotation.hadamard(self.a)).float())
        self.register_buffer("hb", torch.from_numpy(rotation.hadamard(self.b)).float())

    def forward(self, x):
        shape = x.shape
        u = (x * self.d.to(x.dtype)).reshape(*shape[:-1], self.m, self.a, self.b)
        return (torch.matmul(self.ha.to(u.dtype), u) @ self.hb.to(u.dtype)).reshape(shape)


class _ActQ(torch.autograd.Function):
    """v = round(x / s_x) + e (or x / s_x + e unrounded) on live rows and x / s_x on the sink row (position 0, as
    `Served.base` has it), straight through: dv/dx = 1 / s_x. In place where it can be: on the CPU each full pass over
    a call's input is a large part of a 70B forward."""

    @staticmethod
    def forward(ctx, x, sx, e, rnd=True):
        u = x.float().div_(sx)
        ctx.save_for_backward(sx)
        v = (torch.round(u) if rnd else u.clone()).add_(e)
        v[:, 0] = u[:, 0]
        return v.to(torch.bfloat16)

    @staticmethod
    def backward(ctx, g):
        (sx,) = ctx.saved_tensors
        return (g.float() / sx).to(g.dtype), None, None, None


class Served:
    """How a linear call's input is served, per matrix (`MODES`, `set_served`).

    - a16: bf16 (W8A16).
    - a8: int8 per token, symmetric at ±126.
    - e2: HT-PROTO-U's design U. int8 per token at ±126 plus raw E uniform on {−2, −1, 0, 1}, one E per call, served
      as (x + E)·qᵀ + ½·Σq, which folds E's mean in per output.
    - e2f: e2's E on the unrounded x / s_x (in bf16, as a16's activations): E without int8 activations.
    - e1: int8 plus ⌊E/2⌋, uniform on {−1, 0} (b_E = 1), from e2's own draw; its mean is e2's, folded in the same way.
    - e3: int8 plus E uniform on {−1, 0, 1} (a = 3), mean 0, from a stream of its own.

    In every mode but a16 the sink row (position 0) is served exact. Linears that read the same input tensor (q, k and
    v; gate and up) are one call and share its codes and E. Whenever any matrix reads an E, every call draws it,
    whatever its own matrices' modes, so a matrix's E doesn't depend on how the others are served. E is drawn (`draw`)
    from one generator per device, seeded seed + 1000·index (e3's offset by 2,000,000), which `reseed` restarts; a
    model on one GPU, or on the CPU, draws from seed itself.

    With wnoise f (HT-DEFENCE's candidate 3), q, k and v are served as q + F, F live integer weight noise uniform on
    [−f, f] drawn on every call, clipped to ±127 (`hardness_public/rotated.py --live-noise`), from generators of their
    own, so E's draws are the same as without it."""

    def __init__(self, mode: str = "a16", wnoise: int = 0):
        assert mode in MODES, mode
        self.mode, self.wnoise, self.seed, self.gens, self._last = mode, wnoise, 0, {}, None
        self.draws = {MODES[mode][0]} - {None}

    def reseed(self, seed: int, device=None):
        self.seed, self.gens, self._last = seed, {}, None

    def gen(self, device, stream: int = 0) -> torch.Generator:
        g = self.gens.get((device, stream))
        if g is None:
            g = self.gens[(device, stream)] = torch.Generator(device=device)
            g.manual_seed(self.seed + 1000 * (device.index or 0) + 1_000_000 * stream)
        return g

    def noised(self, q: torch.Tensor) -> torch.Tensor:
        f = self.wnoise
        n = torch.randint(-f, f + 1, q.shape, device=q.device, dtype=q.dtype, generator=self.gen(q.device, 1))
        return (q + n).clamp(-127, 127)

    def base(self, x, rot):
        """One call's rotated input, its per-token scale, the live rows and every E the matrices read, drawn once."""
        if self._last is not None and self._last[0] is x and self._last[1] is rot:
            return self._last[2]
        assert x.dim() == 3, x.shape
        xr = x.to(torch.bfloat16)
        if rot is not None:
            xr = rot(xr)
        sx = (xr.detach().abs().amax(-1, keepdim=True).float() / 126).clamp_min(1e-12)
        live = torch.ones(*x.shape[:-1], 1, dtype=torch.bool, device=x.device)
        live[:, 0] = False
        es = {}
        if "e4" in self.draws:
            es["e4"] = draw("e4", xr.shape, self.gen(x.device), x.device)
        if "e3" in self.draws:
            es["e3"] = draw("e3", xr.shape, self.gen(x.device, 2), x.device)
        b = {"xr": xr, "sx": sx, "live": live, "e": es, "v": {}}
        self._last = (x, rot, b)
        return b

    def act(self, x, rot, mode: str | None = None):
        """(v, s_x, the offset that folds E's mean in, or None) for x served as mode (default the base mode)."""
        mode = mode or self.mode
        b = self.base(x, rot)
        if mode not in b["v"]:
            kind, rnd, mean = MODES[mode]
            e = torch.zeros((), device=x.device) if kind is None else b["e"][kind]
            if mode == "e1":
                e = torch.floor(e / 2)
            v = _ActQ.apply(b["xr"], b["sx"], e, rnd)
            b["v"][mode] = (v, b["sx"], mean * b["live"].float() if mean else None)
        return b["v"][mode]


class Probe:
    """Where served noise lands, per matrix, over the live rows (every position but the sink) of its eval calls: the
    signal Σ‖y‖² of its a16 output (low-rank part included) and, per mode, the output error Σ‖y_m − y_a16‖²; the input
    Σ‖x·R‖² and, per mode, its error Σ‖(v_m + mean − x·R/s_x)·s_x‖²; and the rotated input's crest factor per row
    (absmax over RMS: sum, sum of squares, max). `quant.PROBE = Probe(model)` turns it on. It draws nothing, so what is
    served is what would be served without it; a mode whose E the spec doesn't draw is skipped, and weight noise left out."""

    def __init__(self, model, modes=("a8", "e2f", "e2")):
        self.names = {id(getattr(p, a)): n for n, p, a in block_linears(model)}
        self.modes, self.acc = modes, {}

    @torch.no_grad()
    def see(self, m, x, q, s, y, lr):
        rec = self.acc.setdefault(self.names[id(m)], {"rows": 0, "signal": 0.0, "input": 0.0, "crest": [0.0, 0.0, 0.0],
                                                      "out": {}, "in": {}})
        ref = y if m.mode == "a16" else m.product(x, q, s, "a16")
        rec["signal"] += float((ref if lr is None else ref + lr)[:, 1:].double().pow(2).sum())
        sv = m.served
        if sv is None:
            return
        b = sv.base(x, m.rot)
        xr, sx = b["xr"][:, 1:].float(), b["sx"][:, 1:]
        cr = xr.abs().amax(-1) / xr.pow(2).mean(-1).sqrt().clamp_min(1e-30)
        rec["rows"] += cr.numel()
        rec["input"] += float(xr.double().pow(2).sum())
        rec["crest"] = [rec["crest"][0] + float(cr.sum()), rec["crest"][1] + float(cr.double().pow(2).sum()),
                        max(rec["crest"][2], float(cr.max()))]
        for md in self.modes:
            kind, _rnd, mean = MODES[md]
            if kind is not None and kind not in sv.draws:
                continue
            ym = y if md == m.mode and not (sv.wnoise and m.noised) else m.product(x, q, s, md, wnoise=False)
            rec["out"][md] = rec["out"].get(md, 0.0) + float((ym - ref)[:, 1:].double().pow(2).sum())
            v = sv.act(x, m.rot, md)[0][:, 1:].float()
            rec["in"][md] = rec["in"].get(md, 0.0) + float(((v + mean - xr / sx) * sx).double().pow(2).sum())

    def summary(self) -> dict[str, dict]:
        """Per matrix its sums, and per mode the output's and the input's error relative to their signal."""
        out = {}
        for name, r in self.acc.items():
            n = max(r["rows"], 1)
            out[name] = r | {"crest_mean": r["crest"][0] / n, "crest_rms": (r["crest"][1] / n) ** 0.5,
                             "crest_max": r["crest"][2],
                             "out_rel": {md: e / r["signal"] for md, e in r["out"].items()},
                             "in_rel": {md: e / r["input"] for md, e in r["in"].items()}}
        return out


PROBE: Probe | None = None


class QLinear(nn.Module):
    def __init__(self, lin: nn.Linear, c: float | None, rot: Rotation | None = None):
        super().__init__()
        assert lin.bias is None
        self.weight = lin.weight
        self.c = c
        self.rot = rot
        self.served, self.mode, self.noised = None, "a16", False
        self._frozen = None
        self.register_parameter("lora_a", None)
        self.register_parameter("lora_b", None)

    def train(self, mode: bool = True):
        self._frozen = None
        return super().train(mode)

    @torch.no_grad()
    def latent(self) -> torch.Tensor:
        """The latent weight the codes are of: W, or W + B·A (formed in fp32, as `_FQLoRA` forms it)."""
        if self.lora_a is None:
            return self.weight.detach()
        with torch.autocast(self.weight.device.type, enabled=False):
            return torch.addmm(self.weight, self.lora_b, self.lora_a)

    def frozen(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The served codes and scales (int8-valued bf16, fp32), from the current latent weight."""
        if self._frozen is None:
            w = self.latent()
            s = row_scale(w, self.c)
            self._frozen = (codes(w, s).to(torch.bfloat16), s)
        return self._frozen

    def product(self, x, q, s, mode: str, wnoise: bool = True):
        """x·(q·s)ᵀ with x served as mode (`Served.act`), in fp32."""
        sv = self.served
        if sv is None or mode == "a16":
            x = x.to(torch.bfloat16)
            if self.rot is not None:
                x = self.rot(x)
            return F.linear(x, q).float() * s
        v, sx, off = sv.act(x, self.rot, mode)
        noised = wnoise and sv.wnoise and self.noised
        if noised:
            q = sv.noised(q)
        y = F.linear(v, q).float()
        if off is not None:
            y = y + off * (q.float().sum(1) if noised else self.code_sums(q))
        return y * sx * s

    def code_sums(self, q) -> torch.Tensor:
        return q.float().sum(1)

    def lowrank(self, x) -> torch.Tensor | None:
        return None

    def forward(self, x):
        if self.training and self.lora_a is not None:
            q, s = _FQLoRA.apply(self.weight, self.lora_a, self.lora_b, self.c)
        elif self.training:
            s = row_scale(self.weight, self.c)
            q = _FQ.apply(self.weight.float(), s)
        else:
            q, s = self.frozen()
        sv = self.served
        if sv is not None and sv.draws:
            sv.base(x, self.rot)  # draw E whatever this matrix's mode, so every other call's noise is the same
        y = self.product(x, q, s, self.mode)
        lr = self.lowrank(x)
        if PROBE is not None and not self.training:
            PROBE.see(self, x, q, s, y, lr)
        return y if lr is None else y + lr


class FrozenQLinear(QLinear):
    """A QLinear for evaluation only: its codes and scales computed once (`wrap` with frozen) and no latent weight, so a
    model loaded in bf16 holds only the codes."""

    def __init__(self, q: torch.Tensor, s: torch.Tensor, c: float | None, rot: Rotation | None = None):
        nn.Module.__init__(self)
        self.register_buffer("q", q)
        self.register_buffer("s", s)
        self.c, self.rot, self.served, self.mode, self.noised = c, rot, None, "a16", False

    def train(self, mode: bool = True):
        assert not mode, "a FrozenQLinear has no latent weight to train"
        return nn.Module.train(self, mode)

    def frozen(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q, self.s

    def code_sums(self, q) -> torch.Tensor:
        if q is not self.q:
            return q.float().sum(1)
        if getattr(self, "_sums", None) is None:
            self._sums = q.float().sum(1)
        return self._sums


class DeflatedQLinear(FrozenQLinear):
    """A FrozenQLinear whose codes are of W·R less its top-r SVD part A·Bᵀ (`hardness_public/rotated.py --deflate`),
    which is served apart as its own product: exact, in fp32 from the bf16 rotated input, with no noise."""

    def __init__(self, q, s, c, rot, a: torch.Tensor, b: torch.Tensor):
        super().__init__(q, s, c, rot)
        self.register_buffer("a", a)
        self.register_buffer("b", b)

    def lowrank(self, x):
        xr = x.to(torch.bfloat16)
        if self.rot is not None:
            xr = self.rot(xr)
        return (xr.float() @ self.b) @ self.a.T


def top_part(w: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """w's top-r SVD part as A·Bᵀ: A = U_r·Σ_r (n×r) and B = V_r (K×r), from the eigenvectors of wᵀw in float64."""
    w = w.double()
    _evals, v = torch.linalg.eigh(w.T @ w)
    b = v[:, -r:].flip(-1)
    return w @ b, b


def rotate_t(w: torch.Tensor) -> torch.Tensor:
    """`rotation.rotate` on w's device: W·R in float64."""
    n, k = w.shape
    m, a, b = rotation.factors(k)
    d = torch.from_numpy(rotation.signs(k)).to(w.device)
    ha, hb = (torch.from_numpy(rotation.hadamard(x)).to(w.device) for x in (a, b))
    return (ha @ (w.double() * d).reshape(n, m, a, b) @ hb).reshape(n, k)


def block_linears(model) -> list[tuple[str, nn.Module, str]]:
    """(name, parent module, attribute) of every projection in every decoder layer."""
    out = []
    for i, layer in enumerate(model.model.layers):
        for blk in ("self_attn", "mlp"):
            parent = getattr(layer, blk)
            for p in PROJ:
                if hasattr(parent, p):
                    out.append((f"model.layers.{i}.{blk}.{p}.weight", parent, p))
    return out


def trim_heap():
    """Hand glibc's freed heap pages back to the OS (malloc_trim). With MALLOC_MMAP_MAX_=0 (hardness_defence/cpu70b.sh),
    a frozen CPU wrap's float64 temporaries otherwise stay resident between the codes: 128 GB on 70B."""
    libc = ctypes.CDLL(None)
    if hasattr(libc, "malloc_trim"):
        libc.malloc_trim(0)


def wrap(model, target: float | None, rotate: bool = False, log=print, targets: dict | None = None,
         frozen: bool = False, calib: dict | None = None) -> dict[str, dict]:
    """Replace every block linear by a QLinear calibrated to its target, targets[name] where listed (None is absmax) and
    target elsewhere, its weight rotated first with rotate; returns per matrix its c and overflow. A targets value bf16
    leaves the matrix unquantized, the model's own bf16 linear.

    With frozen, each is a FrozenQLinear whose codes come from W·R formed in float64 on the weight's device and then
    rounded to fp32, as the latent path holds it, and overwrite the bf16 weight in place (codes within ±127 are exact in
    bf16; on the CPU, a new tensor in its place): a bf16 model too large for fp32, wrapped in the memory it was loaded
    in plus one matrix's temporaries. A
    targets value deflR (frozen only) is a DeflatedQLinear at target: the codes of W·R less its top-R SVD part. With
    calib (frozen only; an earlier wrap's return on the same weights and targets), each matrix takes its recorded c
    instead of solving for it, and its overflow at that c must be within CALIB_TOL of the one recorded (another
    device's float64 rounds a few residuals to the other side of a half step)."""
    info, rots, targets, drift = {}, {}, targets or {}, 0.0
    assert calib is None or frozen, "calib reuses a frozen wrap's c"
    for name, parent, attr in block_linears(model):
        lin = getattr(parent, attr)
        tgt = targets.get(name, target)
        if tgt == "bf16":
            info[name] = {"target": "bf16"}
            continue
        deflate = 0
        if isinstance(tgt, str):
            assert tgt.startswith("defl") and frozen, f"{name}: {tgt} isn't deflR, or the model isn't frozen"
            deflate, tgt = int(tgt.removeprefix("defl")), target
        dev = lin.weight.device
        rot = None
        if rotate:
            k = lin.weight.shape[1]
            rot = rots.get((k, dev))
            if rot is None:
                rot = rots[(k, dev)] = Rotation(k).to(dev)
        if frozen:
            with torch.no_grad():
                w = rotate_t(lin.weight) if rotate else lin.weight.double()
                if deflate:
                    a, b = top_part(w, deflate)
                    w = w - a @ b.T
                w = w.float()
                if calib is None:
                    c, ov = calibrate(w, tgt)
                else:
                    rec = calib[name]
                    assert (rec["target"], rec.get("deflate", 0)) == (tgt, deflate), f"{name}: calib {rec}, now {tgt}"
                    c, ov = rec["c"], strassen1(w, rec["c"])
                    drift = max(drift, abs(ov - rec["strassen1"]))
                    assert drift <= CALIB_TOL, f"{name}: overflow {ov} at the recorded c, recorded {rec}"
                s = row_scale(w, c)
                q = lin.weight.detach()
                assert q.dtype == torch.bfloat16, q.dtype
                if q.device.type == "cpu":
                    q = codes(w, s).to(torch.bfloat16)  # a CPU model's weights can be views of the checkpoint's mmap
                else:
                    q.copy_(codes(w, s))
            del w
            if dev.type == "cpu":
                trim_heap()
            setattr(parent, attr, DeflatedQLinear(q, s, c, rot, a.float(), b.float()) if deflate
                    else FrozenQLinear(q, s, c, rot))
        else:
            if rotate:
                with torch.no_grad():
                    lin.weight.copy_(torch.from_numpy(rotation.rotate(lin.weight.detach().cpu().numpy())))
            c, ov = calibrate(lin.weight, tgt)
            setattr(parent, attr, QLinear(lin, c, rot))
        info[name] = {"c": c, "strassen1": ov, "target": tgt} | ({"deflate": deflate} if deflate else {})
    ovs = sorted(v["strassen1"] for v in info.values() if "strassen1" in v)
    log(f"wrap: {len(info)} linears ({len(ovs)} quantized), target {target} ({len(targets)} set apart), rotate {rotate}"
        f"{f', calibration reused (overflow within {drift:.2e} of its record)' if calib else ''}, strassen1 median "
        f"{ovs[len(ovs) // 2]:.4%}")
    return info


def selects(sel: str, name: str) -> bool:
    """Whether sel names the matrix: a projection (q_proj), one matrix (layers.0.self_attn.q_proj), every matrix of
    layers A to B (layers.A-B), or one projection of each (layers.A-B.q_proj)."""
    short = name.removeprefix("model.").removesuffix(".weight")
    if sel in (short, short.rsplit(".", 1)[-1]):
        return True
    head, _, proj = sel.removeprefix("layers.").partition(".")
    lo, dash, hi = head.partition("-")
    if not (sel.startswith("layers.") and dash and lo.isdigit() and hi.isdigit()):
        return False
    return int(lo) <= int(short.split(".")[1]) <= int(hi) and proj in ("", short.rsplit(".", 1)[-1])


def set_served(model, spec: str) -> Served:
    """Serve every QLinear as spec, MODE[/wF][:X+X...][;MODE=X+X...]... (`Served`, `MODES`): /wF adds live weight noise
    uniform on [−F, F] to q, k and v; each X selects matrices (`selects`). Each ;MODE=X+... serves its matrices as
    MODE, later ones over earlier ones, and the X after the colon are served exact (a16: activations in bf16, no
    noise) over all of them."""
    head, *over = spec.split(";")
    head, _, exact = head.partition(":")
    mode, _, w = head.partition("/")
    assert not w or w.startswith("w"), spec
    rules = [(m, sels.split("+")) for m, _, sels in (o.partition("=") for o in over)]
    assert all(m in MODES and all(sels) for m, sels in rules), spec
    rules.append(("a16", [x for x in exact.split("+") if x]))
    names = [n for n, _p, _a in block_linears(model)]
    unmatched = [x for _m, sels in rules for x in sels if not any(selects(x, n) for n in names)]
    assert not unmatched, f"{spec}: {unmatched} select no matrix"
    sv = Served(mode, int(w[1:]) if w else 0)
    sv.draws = set()
    for name, parent, attr in block_linears(model):
        m = getattr(parent, attr)
        if not isinstance(m, QLinear):
            continue
        md = mode
        for rm, sels in rules:
            if any(selects(x, name) for x in sels):
                md = rm
        m.served, m.mode = sv, md
        m.noised = attr in ("q_proj", "k_proj", "v_proj")
        sv.draws |= {MODES[md][0]} - {None}
    return sv


def recalibrate(model, target: float | None) -> dict[str, dict]:
    info = {}
    for name, parent, attr in block_linears(model):
        m = getattr(parent, attr)
        if not isinstance(m, QLinear):
            continue
        if target is not None:
            m.c, ov = calibrate(m.latent(), target)
        else:
            ov = strassen1(m.latent(), None)
        m._frozen = None
        info[name] = {"c": m.c, "strassen1": ov}
    return info


def add_lora(model, r: int, seed: int = 0) -> int:
    """Freeze every parameter and give every QLinear a trained rank-r update of its latent weight, B·A, with A Gaussian
    at 1/√K and B zero, so the codes start as they were; returns the number of parameters added."""
    for p in model.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(seed)
    added = 0
    for _name, parent, attr in block_linears(model):
        m = getattr(parent, attr)
        if type(m) is not QLinear:
            continue
        n, k = m.weight.shape
        m.lora_a = nn.Parameter((torch.randn(r, k, generator=g) / math.sqrt(k)).to(m.weight.device))
        m.lora_b = nn.Parameter(torch.zeros(n, r, device=m.weight.device))
        added += r * (n + k)
    return added


@torch.no_grad()
def merge_lora(model) -> int:
    """Fold each QLinear's B·A into its latent weight (`QLinear.latent`'s sum) and drop A and B; returns how many."""
    merged = 0
    for _name, parent, attr in block_linears(model):
        m = getattr(parent, attr)
        if type(m) is QLinear and m.lora_a is not None:
            m.weight.copy_(m.latent())
            m.lora_a = m.lora_b = None
            m._frozen = None
            merged += 1
    return merged


def stats(model) -> dict[str, dict]:
    """Per matrix: c, the overflow profile to level 2, σ of the codes and the saturated share."""
    out = {}
    with torch.no_grad():
        for name, parent, attr in block_linears(model):
            m = getattr(parent, attr)
            if not isinstance(m, QLinear):
                continue
            q, _s = m.frozen()
            prof = overflow(q, 2)
            out[name] = {"shape": list(q.shape), "c": m.c, "sigma_q": float(q.float().std()),
                         "saturated": float((q.abs() == 127).float().mean()), "overflow": prof}
    return out
