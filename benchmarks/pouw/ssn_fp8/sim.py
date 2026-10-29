"""Vectorized GroupSum: `verity.ml.circuits.silicon.models.GroupSum.step` on numpy arrays, bit for bit.

One call evaluates many independent tensor-core steps at once: ``acc`` holds FP32 words of any shape ``S``, ``a`` and
``b`` hold operand words of shape ``S + (k,)``.  The semantics are the pinned model's (products exact, every term
truncated toward zero to the grid of the largest exponent, an exact sum, the result truncated to ``width`` bits and
back to FP32); `check` holds it to the scalar model on random words.  Non-finite operands, non-finite accumulators and
results beyond the FP32 range are outside the finite model and come back flagged in ``bad``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from verity.ml.circuits.silicon import models as M
from verity.ml.circuits.silicon.term import FORMATS

FP32_MIN_NORMAL_EXPONENT = -126
FP32_MAX_EXPONENT = 127


@dataclass(frozen=True)
class Pipe:
    dtype: str
    groups: tuple[int, ...]
    width: int
    zero_exponent: int

    @classmethod
    def of(cls, model: M.GroupSum) -> "Pipe":
        return cls(model.dtype, tuple(model.groups), model.width, model.zero_exponent)

    @property
    def k(self) -> int:
        return sum(self.groups)


E4M3_K32 = Pipe.of(M.HOPPER_E4M3_K32)
E5M2_K32 = Pipe.of(M.HOPPER_E5M2_K32)
BF16_K16 = Pipe.of(M.HOPPER_BF16_WGMMA_K16)


def products(dtype: str, a: np.ndarray, b: np.ndarray):
    """(negative, exponent, magnitude, nonfinite) of the exact products, as `term.Format.product` carries them."""
    f = FORMATS[dtype]
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    mm, me = (1 << f.man_bits) - 1, (1 << f.exp_bits) - 1
    ea, eb = (a >> f.man_bits) & me, (b >> f.man_bits) & me
    if f.nan_rule == "ieee":
        bad = (ea == me) | (eb == me)
    elif f.nan_rule == "e4m3":
        bad = ((a & 0x7F) == 0x7F) | ((b & 0x7F) == 0x7F)
    else:
        bad = np.zeros(a.shape, bool)
    hidden = 1 << f.man_bits
    sa = (a & mm) | np.where(ea > 0, hidden, 0)
    sb = (b & mm) | np.where(eb > 0, hidden, 0)
    mag = (sa * sb) << f.product_shift
    exp = np.maximum(ea, 1) + np.maximum(eb, 1) + f.product_exponent_offset
    neg = (((a ^ b) >> (f.bits - 1)) & 1) == 1
    return neg, exp, np.where(bad, 0, mag), bad


def fp32_terms(c: np.ndarray):
    c = c.astype(np.int64) & 0xFFFFFFFF
    field = (c >> 23) & 0xFF
    mant = c & 0x7FFFFF
    bad = field == 0xFF
    exp = np.where(field == 0, FP32_MIN_NORMAL_EXPONENT, field - 127)
    mag = np.where(field == 0, mant, mant | (1 << 23))
    return (c >> 31) == 1, exp, np.where(bad, 0, mag), bad


def _bit_length(m: np.ndarray) -> np.ndarray:
    _, e = np.frexp(m.astype(np.float64))
    return np.where(m > 0, e, 0).astype(np.int64)


def group_sum(neg, exp, mag, width: int, zero_exponent: int):
    """`models.group_sum` along the last axis: (negative, exponent, magnitude) of the 24-bit result."""
    rescale = width - 24
    nz = mag > 0
    max_exp = np.maximum(np.max(np.where(nz, exp, zero_exponent), axis=-1), zero_exponent)
    scaled = mag << rescale if rescale >= 0 else mag >> -rescale
    shift = np.clip(max_exp[..., None] - exp, 0, 62)
    aligned = np.where(nz, scaled >> shift, 0)
    total = np.sum(np.where(neg, -aligned, aligned), axis=-1)
    rneg = total < 0
    m = np.abs(total)
    bl = _bit_length(m)
    e = max_exp + bl - width
    m = np.where(bl > width, m >> np.clip(bl - width, 0, 62), m << np.clip(width - bl, 0, 62))
    sub = e < FP32_MIN_NORMAL_EXPONENT
    m = np.where(sub, m >> np.clip(FP32_MIN_NORMAL_EXPONENT - e, 0, 62), m)
    e = np.where(sub, FP32_MIN_NORMAL_EXPONENT, e)
    m = m >> rescale if rescale >= 0 else m << -rescale
    return rneg & (m > 0), e, m


def pack(neg, exp, mag):
    """FP32 words (zero is +0); also the overflow flag (exponent beyond 127, packed as an infinity)."""
    over = (mag > 0) & (exp > FP32_MAX_EXPONENT)
    field = exp + 127 - np.where(mag & (1 << 23), 0, 1)
    words = (np.where(neg, 1, 0) << 31) | (np.clip(field, 0, 255) << 23) | (mag & 0x7FFFFF)
    words = np.where(over, (np.where(neg, 1, 0) << 31) | 0x7F800000, words)
    return np.where(mag > 0, words, 0).astype(np.uint32), over


def step(p: Pipe, acc: np.ndarray, a: np.ndarray, b: np.ndarray):
    """One instruction step: FP32 words out, and the out-of-domain flag."""
    if a.shape[-1] != p.k or b.shape[-1] != p.k:
        raise ValueError(f"{p.dtype} step takes {p.k} operand pairs")
    pn, pe, pm, pbad = products(p.dtype, a, b)
    cn, ce, cm, cbad = fp32_terms(acc)
    bad = cbad | pbad.any(-1)
    start = 0
    over = np.zeros(acc.shape, bool)
    for size in p.groups:
        sl = slice(start, start + size)
        neg = np.concatenate([cn[..., None], pn[..., sl]], -1)
        exp = np.concatenate([ce[..., None], pe[..., sl]], -1)
        mag = np.concatenate([cm[..., None], pm[..., sl]], -1)
        cn, ce, cm = group_sum(neg, exp, mag, p.width, p.zero_exponent)
        over |= (cm > 0) & (ce > FP32_MAX_EXPONENT)
        start += size
    words, o = pack(cn, ce, cm)
    return words, bad | over | o


def chain(p: Pipe, acc: np.ndarray, a: np.ndarray, b: np.ndarray, keep: bool = True):
    """Fold `step` over k slices of a, b (last axis a multiple of p.k); every step's word (axis -1) if ``keep``."""
    K = a.shape[-1]
    if K % p.k:
        raise ValueError(f"K = {K} is not a multiple of {p.k}")
    words, bad = [], np.zeros(acc.shape, bool)
    for s in range(0, K, p.k):
        acc, bb = step(p, acc, a[..., s:s + p.k], b[..., s:s + p.k])
        bad |= bb
        if keep:
            words.append(acc)
    return (np.stack(words, -1) if keep else acc), bad


def fp32_value(words: np.ndarray) -> np.ndarray:
    return np.asarray(words, np.uint32).view(np.float32).astype(np.float64)


def check(n: int = 4000, seed: int = 0) -> dict:
    """The vectorized step against the scalar pinned model on random finite words (and random accumulators)."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, model in (("e4m3", M.HOPPER_E4M3_K32), ("e5m2", M.HOPPER_E5M2_K32), ("bf16", M.HOPPER_BF16_WGMMA_K16)):
        p = Pipe.of(model)
        bits = 16 if name == "bf16" else 8
        a = rng.integers(0, 1 << bits, (n, p.k))
        b = rng.integers(0, 1 << bits, (n, p.k))
        expo = rng.integers(-40, 40, n)
        c = (rng.standard_normal(n) * np.exp2(expo)).astype(np.float32).view(np.uint32)
        c[: n // 8] = 0
        got, bad = step(p, c, a, b)
        mism = checked = 0
        for i in range(n):
            if bad[i]:
                continue
            checked += 1
            mism += int(model.step(int(c[i]), a[i].tolist(), b[i].tolist()) != int(got[i]))
        out[name] = {"checked": checked, "mismatches": mism}
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(check()))
