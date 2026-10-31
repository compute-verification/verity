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
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class QLinear(nn.Module):
    def __init__(self, lin: nn.Linear, c: float | None):
        super().__init__()
        assert lin.bias is None
        self.weight = lin.weight
        self.c = c
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
        return F.linear(x.to(torch.bfloat16), q).float() * s


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


def wrap(model, target: float | None, log=print) -> dict[str, dict]:
    """Replace every block linear by a QLinear calibrated to target; returns per matrix its c and overflow."""
    info = {}
    for name, parent, attr in block_linears(model):
        lin = getattr(parent, attr)
        c, ov = calibrate(lin.weight, target)
        setattr(parent, attr, QLinear(lin, c))
        info[name] = {"c": c, "strassen1": ov}
    log(f"wrap: {len(info)} linears, target {target}, strassen1 median "
        f"{sorted(v['strassen1'] for v in info.values())[len(info) // 2]:.4%}")
    return info


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
