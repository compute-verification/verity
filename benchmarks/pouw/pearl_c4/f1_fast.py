"""Pearl-C4's F1′ for the verifier, vectorised (numpy), beside the reference's pure-Python rule (`pearl_c4.tile_debit`).

`row_law` is the exact noise law of a row, the law `pearl_c4.noise_law` convolves, by one FFT per row sized to the row's
own support: n = Σ_r (2E_r)(2F_r), F uniform over the 16 E2M1 codes, so its characteristic function is a product of
powers of one draw's, per |E| magnitude.  It agrees with the reference to float rounding (about 1e-16; a test).

`split_units_fast` is C (the assessor's B), which is
now the rule itself (`pearl_c4.change_probabilities`, normative for prover and verifier): this is the same computation,
vectorised, and agrees with the reference to float rounding (a test), inside R1's tolerance band.
- The modal byte by a certified local search: from the product formula at the four edges around the realised byte r,
  P(r − 1), P(r) and P(r + 1) are known with the mass outside them; when the largest of the three exceeds that mass no
  other byte can win, so it is the modal byte exactly.  Uncertified blocks retry at ±4 and ±16; a block still uncertified
  is charged on the safe side: p = 0 on its codes and no κ relief.
- p_t from 4 lookups, a lower bound: the cells beside the element's own resolve three cells exactly, and each tail
  beyond them counts as one cell, so 1 − max(three cells, two tails) ≤ p_t.  S_w only grows as any p_t falls, at the
  sorted in-block layout.  A 1e-12 slack covers the FFT's rounding.
- κ inside, the sorted layout, the truncated DPs and the cap at 1 are the reference's (`pearl_c4.split_saving`).
Salt-dead codes take p = 0 from the caller's witness (the verifier already computes it for the forming item)."""
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np
from verity.core.protocols.pouw.schemes import pearl_c4 as F
from verity.core.protocols.pouw.schemes import pearl_kw as P

MAGS = (1, 2, 3, 4, 6, 8, 12)                      # |2·value| of the nonzero E2M1 magnitudes
SLACK = F.F1_EPS
C = F.FFMA_UNITS / F.SPAN_K
XM = 4                                              # ½ − c·x < 0 from x = 4 on
_GAIN = np.array([max(0.0, 0.5 - C * x) for x in range(XM + 1)])
_EDGES = np.array([-m for m in reversed(F._E2M1_MIDPOINTS)] + list(F._E2M1_MIDPOINTS))
_COMBOS = np.array(list(itertools.product(range(3), repeat=4)))
_EXCESS = np.array([F._pair_excess(tuple(js)) for js in _COMBOS])


def _phi(size: int) -> np.ndarray:
    w = 2 * np.pi * np.arange(size // 2 + 1) / size
    return np.stack([(2 + 2 * sum(np.cos(w * v * h) for h in MAGS)) / 16 for v in MAGS])


def row_law(e_row: Sequence[int]) -> tuple[int, np.ndarray]:
    """(lo, pmf) of n given the E line, as `pearl_c4.noise_law`: the pmf over the consecutive integers from lo."""
    e2 = np.abs(np.array([F._F2[c] for c in e_row]))
    L = int(12 * e2.sum())
    if L == 0:
        return 0, np.ones(1)
    size = 1 << int(math.ceil(math.log2(2 * L + 1)))
    ph = _phi(size)
    cf = np.ones(size // 2 + 1)
    for i, v in enumerate(MAGS):
        m = int((e2 == v).sum())
        if m:
            cf *= ph[i] ** m
    pmf = np.fft.irfft(cf, size)[np.arange(-L, L + 1) % size]
    return -L, np.maximum(pmf, 0.0)


def _cdf(lo: int, pmf: np.ndarray, scale: float):
    """y ↦ P(noise < y) for noise = scale·n, vectorised: cum[ceil(y/scale) − lo], clamped."""
    cum = np.concatenate([[0.0], np.cumsum(pmf)])
    cum /= cum[-1]

    def f(y):
        y = np.asarray(y, dtype=np.float64)
        t = np.where(np.isfinite(y), np.ceil(np.divide(y, scale, where=np.isfinite(y), out=np.zeros_like(y))) - lo,
                     np.where(y > 0, len(cum) - 1, 0))
        return cum[np.clip(t, 0, len(cum) - 1).astype(np.int64)]
    return f


def _byte_value(s):
    s = np.asarray(s)
    e, m = (s & 0x7F) >> 3, s & 7
    return np.where(e > 0, (8 + m) * 2.0 ** (e - 10.0), m * 2.0 ** -9)


def _edges_between(lo_b: np.ndarray, hi_b: np.ndarray, width: int) -> np.ndarray:
    """Per block, the amax edges of bytes lo_b … lo_b + width − 1 (clamped to 0x01 … 0x7E): width + 1 edges."""
    s = lo_b[:, None] + np.arange(width + 1)[None, :]          # edge j sits below byte lo_b + j
    below = np.where(s - 1 >= F.UE4M3_MIN, 3 * (_byte_value(np.clip(s - 1, 1, 0x7E)) + _byte_value(np.clip(s, 1, 0x7E))), 0.0)
    below = np.where(s <= F.UE4M3_MIN, 0.0, below)
    return np.where(s > F.UE4M3_TOP, np.inf, below)


def certified_modes(v: np.ndarray, realised: np.ndarray, cdf) -> np.ndarray:
    """Per block of v (blocks × 16), its modal byte, certified by a local search around the realised byte, or −1."""
    nb = len(v)
    out = np.full(nb, -1, dtype=np.int64)
    todo = np.arange(nb)
    for w in (1, 4, 16):
        if not len(todo):
            break
        r = realised[todo]
        lo_b = np.clip(r - w, F.UE4M3_MIN, F.UE4M3_TOP)
        hi_b = np.clip(r + w, F.UE4M3_MIN, F.UE4M3_TOP)
        width = int((hi_b - lo_b).max()) + 1
        E = _edges_between(lo_b, hi_b, width)                    # (n, width + 1)
        vv = v[todo][:, None, :]
        U = E[:, :, None]
        below = np.prod(np.clip(cdf(U - vv) - cdf(-U - vv), 0.0, 1.0), axis=2)   # P(amax < edge)
        below = np.where(np.isinf(E), 1.0, below)
        probs = np.diff(below, axis=1)                           # P(byte = lo_b + j)
        valid = (lo_b[:, None] + np.arange(width)[None, :]) <= hi_b[:, None]
        probs = np.where(valid, probs, -1.0)
        last = np.minimum(hi_b - lo_b + 1, width)
        outside = below[:, 0] + 1.0 - below[np.arange(len(todo)), last]
        j = np.argmax(probs, axis=1)                             # first maximum: the smaller byte on ties
        best = probs[np.arange(len(todo)), j]
        rest = probs.copy()
        rest[np.arange(len(todo)), j] = -1.0
        runner = np.maximum(outside, rest.max(axis=1))
        ok = best > runner + SLACK
        out[todo[ok]] = (lo_b + j)[ok]
        todo = todo[~ok]
    return out


def p_lower(v: np.ndarray, modes: np.ndarray, cdf) -> np.ndarray:
    """Per element (blocks × 16), a lower bound on P(code ≠ modal code) at the block's modal byte, from 4 lookups."""
    vs = _byte_value(np.maximum(modes, 1))[:, None]
    edges = _EDGES[None, None, :] * vs[..., None]                # (blocks, 1, 14)
    c = (edges <= v[..., None]).sum(-1)                          # own cell index, 0 … 14
    ext = np.concatenate([np.full(v.shape + (1,), -np.inf), np.broadcast_to(edges, v.shape + (14,)),
                          np.full(v.shape + (1,), np.inf)], axis=-1)   # ext[j] = edge j − 1, j = 0 … 15
    pick = lambda j: np.take_along_axis(ext, np.clip(j, 0, 15)[..., None], -1)[..., 0]  # noqa: E731
    e_m2, e_m1, e_0, e_1 = (pick(c + d) for d in (-1, 0, 1, 2))
    f = [cdf(e - v) for e in (e_m2, e_m1, e_0, e_1)]
    left, own, right = f[1] - f[0], f[2] - f[1], f[3] - f[2]
    tails = np.maximum(f[0], 1.0 - f[3])
    return np.maximum(0.0, 1.0 - np.maximum.reduce([left, own, right, tails]) - SLACK)


def _conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    for i in range(XM + 1):
        for j in range(XM + 1):
            out[..., min(i + j, XM)] += a[..., i] * b[..., j]
    return out


def _chunk(pairs: np.ndarray) -> np.ndarray:
    """pairs (..., 4, 2): each 8-chunk's excess distribution."""
    a, b = pairs[..., 0], pairs[..., 1]
    pj = np.stack([(1 - a) * (1 - b), a * (1 - b) + (1 - a) * b, a * b], -1)      # (..., 4, 3)
    out = np.zeros(pairs.shape[:-2] + (XM + 1,))
    for js, x in zip(_COMBOS, _EXCESS):
        out[..., x] += pj[..., 0, js[0]] * pj[..., 1, js[1]] * pj[..., 2, js[2]] * pj[..., 3, js[3]]
    return out


def _changed(p: np.ndarray) -> np.ndarray:
    d = np.zeros(p.shape[:-1] + (XM + 1,)); d[..., 0] = 1.0
    for t in range(p.shape[-1]):
        q = p[..., t:t + 1]
        nd = d * (1 - q)
        nd[..., 1:] += d[..., :-1] * q
        nd[..., XM] += d[..., XM] * q[..., 0]
        d = nd
    return d


def window_charges(p: np.ndarray, kappa: np.ndarray, block: int = 16) -> np.ndarray:
    """p (windows × 128), kappa (windows,): `pearl_c4.split_saving` per window, vectorised."""
    W = len(p)
    s = np.sort(p.reshape(W, -1, block), axis=-1)
    pairs = s.reshape(W, s.shape[1], block // 2, 2)
    n = block // 8
    d = np.zeros((W, XM + 1)); d[:, 0] = 1.0
    for b in range(s.shape[1]):
        blk = np.zeros((W, XM + 1)); blk[:, 0] = 1.0
        for ch in range(n):
            blk = _conv(blk, _chunk(pairs[:, b, ch::n]))
        d = _conv(d, blk)
    k = kappa[:, None]
    out = (d * np.maximum(0.0, _GAIN[None, :] - k)).sum(1)
    h0, h1 = _changed(p[:, :F.ATOM]), _changed(p[:, F.ATOM:F.SPAN_K])
    g = _GAIN[:, None] + _GAIN[None, :]
    out += np.einsum("wi,wj,wij->w", h0, h1, np.maximum(0.0, g[None] - kappa[:, None, None]))
    return np.minimum(1.0, out)


def split_units_fast(x: Sequence[int], alpha: int, e_row: Sequence[int], ae: int, scales: Sequence[int],
                     dead: Sequence[bool], overfit: float = 1) -> tuple[Fraction, dict]:
    """F1′ per word the row feeds (NVFP4), by the cheap sound evaluation: at least `pearl_c4.split_units` on the exact
    law.  Returns (units, counts)."""
    k = len(x) - len(x) % F.SPAN_K
    a32 = np.float32(P.f32_of_bits(alpha))
    v = (np.array([P.f32_of_bits(w) for w in x[:k]], dtype=np.float32) * a32).astype(np.float64).reshape(-1, 16)
    lo, pmf = row_law(e_row)
    cdf = _cdf(lo, pmf, F._scale_value(F.NVFP4, ae) / 64)
    realised = np.array(F.hw_scales(F.NVFP4, scales[:k // 16]), dtype=np.int64)
    modes = certified_modes(v, realised, cdf)
    uncertified = modes < 0
    p = np.where(uncertified[:, None], 0.0, p_lower(v, modes, cdf))
    p = np.where(np.array(dead[:k]).reshape(-1, 16), 0.0, p).reshape(-1, F.SPAN_K)
    moved = ((realised != modes) & ~uncertified).reshape(-1, F.SPAN_K // 16).sum(1)
    kappa = np.minimum(C * moved, F.DENSE_CORRECTION)
    units = F.SPAN_K * np.minimum(1.0, overfit * window_charges(p, kappa)).sum()
    return Fraction(float(units)), {"blocks": len(modes), "uncertified": int(uncertified.sum())}


def row_units(fmt: F.Fmt, side: F.Formed, e_rows: Sequence[Sequence[int]], pos: int, dead: Sequence[bool],
              overfit: float = 1) -> tuple[Fraction, int]:
    """`pearl_c4.RowF1`, vectorised (NVFP4): `PearlC4(f1=row_units)` makes it the verifier's path."""
    if fmt is not F.NVFP4:
        return F.row_f1(fmt, side, e_rows, pos, dead, overfit)
    units, counts = split_units_fast(side.x[pos], side.alpha[pos], e_rows[pos], side.ae[pos], side.scales[pos], dead,
                                     overfit)
    return units, counts["uncertified"]
