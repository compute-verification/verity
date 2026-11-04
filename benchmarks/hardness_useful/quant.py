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

`Served` is how the activations reach the codes: bf16, int8 per token, or design U's int8 plus live 2-bit noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import rotation

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
S_V = [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
W_V = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 1, -1], [-1, 1, 0, 0], [1, -1, 0, 1], [0, -1, 0, 1]]
SCHEMES = {"strassen": S_V, "winograd": W_V}


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
    """v = round(x / s_x) + e on live rows and x / s_x on the sink row, straight through: dv/dx = 1 / s_x."""

    @staticmethod
    def forward(ctx, x, sx, e, live):
        u = x.float() / sx
        ctx.save_for_backward(sx)
        return torch.where(live, torch.round(u) + e, u).to(torch.bfloat16)

    @staticmethod
    def backward(ctx, g):
        (sx,) = ctx.saved_tensors
        return (g.float() / sx).to(g.dtype), None, None, None


class Served:
    """How a linear call's input is served.

    - a16: bf16 (W8A16).
    - a8: int8 per token, symmetric at ±126.
    - e2: HT-PROTO-U's design U. int8 per token at ±126 plus raw E uniform on {−2, −1, 0, 1}, one E per call, served
      as (x + E)·qᵀ + ½·Σq, which folds E's mean in per output.

    In a8 and e2 the sink row (position 0) is served exact. Linears that read the same input tensor (q, k and v; gate and
    up) are one call and share its codes and E. E is drawn from one generator per GPU, seeded seed + 1000·index, which
    `reseed` restarts; a model on one GPU draws from seed itself.

    With wnoise f (HT-DEFENCE's candidate 3), q, k and v are served as q + F, F live integer weight noise uniform on
    [−f, f] drawn on every call, clipped to ±127 (`hardness_public/rotated.py --live-noise`), from generators of their
    own, so E's draws are the same as without it."""

    def __init__(self, mode: str = "a16", wnoise: int = 0):
        assert mode in ("a16", "a8", "e2"), mode
        self.mode, self.wnoise, self.seed, self.gens, self._last = mode, wnoise, 0, {}, None

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

    def act(self, x, rot):
        if self._last is not None and self._last[0] is x and self._last[1] is rot:
            return self._last[2:]
        assert x.dim() == 3, x.shape
        xr = x.to(torch.bfloat16)
        if rot is not None:
            xr = rot(xr)
        sx = (xr.detach().abs().amax(-1, keepdim=True).float() / 126).clamp_min(1e-12)
        live = torch.ones(*x.shape[:-1], 1, dtype=torch.bool, device=x.device)
        live[:, 0] = False
        if self.mode == "e2":
            e = torch.randint(-2, 2, xr.shape, device=x.device, generator=self.gen(x.device)).float()
            off = 0.5 * live.float()
        else:
            e, off = torch.zeros((), device=x.device), None
        v = _ActQ.apply(xr, sx, e, live)
        self._last = (x, rot, v, sx, off)
        return v, sx, off


class QLinear(nn.Module):
    def __init__(self, lin: nn.Linear, c: float | None, rot: Rotation | None = None):
        super().__init__()
        assert lin.bias is None
        self.weight = lin.weight
        self.c = c
        self.rot = rot
        self.served, self.exact, self.noised = None, False, False
        self._frozen = None

    def train(self, mode: bool = True):
        self._frozen = None
        return super().train(mode)

    def frozen(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The served codes and scales (int8-valued bf16, fp32), from the current latent weight."""
        if self._frozen is None:
            s = row_scale(self.weight, self.c)
            self._frozen = (codes(self.weight, s).to(torch.bfloat16), s)
        return self._frozen

    def forward(self, x):
        if self.training:
            s = row_scale(self.weight, self.c)
            q = _FQ.apply(self.weight.float(), s)
        else:
            q, s = self.frozen()
        if self.served is None or self.served.mode == "a16" or self.exact:
            if self.exact and self.served is not None and self.served.mode == "e2":
                self.served.act(x, self.rot)  # draw E anyway, so every other call's noise is the same as with none exact
            x = x.to(torch.bfloat16)
            if self.rot is not None:
                x = self.rot(x)
            return F.linear(x, q).float() * s
        v, sx, off = self.served.act(x, self.rot)
        if self.served.wnoise and self.noised:
            q = self.served.noised(q)
        y = F.linear(v, q).float()
        if off is not None:
            y = y + off * q.float().sum(1)
        return y * sx * s


class FrozenQLinear(QLinear):
    """A QLinear for evaluation only: its codes and scales computed once (`wrap` with frozen) and no latent weight, so a
    model loaded in bf16 holds only the codes."""

    def __init__(self, q: torch.Tensor, s: torch.Tensor, c: float | None, rot: Rotation | None = None):
        nn.Module.__init__(self)
        self.register_buffer("q", q)
        self.register_buffer("s", s)
        self.c, self.rot, self.served, self.exact, self.noised = c, rot, None, False, False

    def train(self, mode: bool = True):
        assert not mode, "a FrozenQLinear has no latent weight to train"
        return nn.Module.train(self, mode)

    def frozen(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q, self.s


class DeflatedQLinear(FrozenQLinear):
    """A FrozenQLinear whose codes are of W·R less its top-r SVD part A·Bᵀ (`hardness_public/rotated.py --deflate`),
    which is served apart as its own product: exact, in fp32 from the bf16 rotated input, with no noise."""

    def __init__(self, q, s, c, rot, a: torch.Tensor, b: torch.Tensor):
        super().__init__(q, s, c, rot)
        self.register_buffer("a", a)
        self.register_buffer("b", b)

    def forward(self, x):
        y = super().forward(x)
        xr = x.to(torch.bfloat16)
        if self.rot is not None:
            xr = self.rot(xr)
        return y + (xr.float() @ self.b) @ self.a.T


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


def wrap(model, target: float | None, rotate: bool = False, log=print, targets: dict | None = None,
         frozen: bool = False) -> dict[str, dict]:
    """Replace every block linear by a QLinear calibrated to its target, targets[name] where listed (None is absmax) and
    target elsewhere, its weight rotated first with rotate; returns per matrix its c and overflow.

    With frozen, each is a FrozenQLinear whose codes come from W·R formed in float64 on the weight's device and then
    rounded to fp32, as the latent path holds it, and overwrite the bf16 weight in place (codes within ±127 are exact in
    bf16): a bf16 model too large for fp32, wrapped in the memory it was loaded in plus one matrix's temporaries. A
    targets value deflR (frozen only) is a DeflatedQLinear at target: the codes of W·R less its top-R SVD part."""
    info, rots, targets = {}, {}, targets or {}
    for name, parent, attr in block_linears(model):
        lin = getattr(parent, attr)
        tgt = targets.get(name, target)
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
                c, ov = calibrate(w, tgt)
                s = row_scale(w, c)
                q = lin.weight.detach()
                assert q.dtype == torch.bfloat16, q.dtype
                q.copy_(codes(w, s))
            del w
            setattr(parent, attr, DeflatedQLinear(q, s, c, rot, a.float(), b.float()) if deflate
                    else FrozenQLinear(q, s, c, rot))
        else:
            if rotate:
                with torch.no_grad():
                    lin.weight.copy_(torch.from_numpy(rotation.rotate(lin.weight.detach().cpu().numpy())))
            c, ov = calibrate(lin.weight, tgt)
            setattr(parent, attr, QLinear(lin, c, rot))
        info[name] = {"c": c, "strassen1": ov, "target": tgt} | ({"deflate": deflate} if deflate else {})
    log(f"wrap: {len(info)} linears, target {target} ({len(targets)} set apart), rotate {rotate}, strassen1 median "
        f"{sorted(v['strassen1'] for v in info.values())[len(info) // 2]:.4%}")
    return info


def set_served(model, spec: str) -> Served:
    """Serve every QLinear as spec, MODE[/wF][:X+X...] (`Served`): /wF adds live weight noise uniform on [−F, F] to q,
    k and v, and each X is a projection (q_proj) or one matrix (layers.0.self_attn.q_proj), served exact: its
    activations in bf16, no noise."""
    head, _, exact = spec.partition(":")
    mode, _, w = head.partition("/")
    assert not w or w.startswith("w"), spec
    ex = set(exact.split("+")) - {""}
    sv = Served(mode, int(w[1:]) if w else 0)
    for name, parent, attr in block_linears(model):
        m = getattr(parent, attr)
        m.served, m.exact = sv, attr in ex or name.removeprefix("model.").removesuffix(".weight") in ex
        m.noised = attr in ("q_proj", "k_proj", "v_proj")
    return sv


def recalibrate(model, target: float | None) -> dict[str, dict]:
    info = {}
    for name, parent, attr in block_linears(model):
        m = getattr(parent, attr)
        if target is not None:
            m.c, ov = calibrate(m.weight, target)
        else:
            ov = strassen1(m.weight, None)
        m._frozen = None
        info[name] = {"c": m.c, "strassen1": ov}
    return info


def stats(model) -> dict[str, dict]:
    """Per matrix: c, the overflow profile to level 2, σ of the codes and the saturated share."""
    out = {}
    with torch.no_grad():
        for name, parent, attr in block_linears(model):
            m = getattr(parent, attr)
            q, _s = m.frozen()
            prof = overflow(q, 2)
            out[name] = {"shape": list(q.shape), "c": m.c, "sigma_q": float(q.float().std()),
                         "saturated": float((q.abs() == 127).float().mean()), "overflow": prof}
    return out
