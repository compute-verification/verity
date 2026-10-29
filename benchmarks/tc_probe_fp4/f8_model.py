"""Vectorized ``GroupSum`` for the K = 32 FP8 rows: the candidate models a ``tcgen05_f8.cu`` capture is scored
against, evaluated on whole arrays of raw words.

``silicon.models.GroupSum.step`` is the reference: one output at a time in Python.  This module computes the same function on
numpy arrays, so that 10^8 captured outputs can be scored, and generalises it in the two ways the FP8 kinds need:

* mixed operand formats (``kind::f8f6f4`` takes an E4M3 A with an E5M2 B), each product carried as ``term.Format.product``
  carries a same-format one (``k32.mixed_product``'s convention);
* a UE8M0 scale pair (``kind::mxf8f6f4 .scale_vec::1X``): ``(sa - 127) + (sb - 127)`` added to the exponent of every product
  of the step before alignment (``k32.GroupSumCandidate``'s convention, measured on sm_120's QMMA.SF).

``tests/test_f8_model.py`` holds :func:`step` to ``GroupSum.step`` word for word.  Every array is int64; no term can exceed
2^31 at widths up to 30, so a 33-term sum fits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from verity.ml.circuits.silicon.term import FORMATS, PRODUCT_WIDTH

FP32_MIN_NORMAL_EXPONENT = -126
FP32_MAX_EXPONENT = 127
_NEG_INF_EXP = -(1 << 20)

#: format name -> the 3-bit code tcgen05's instruction descriptor gives it (PTX ISA 9.7.18.4, kind::f8f6f4 / kind::mxf8f6f4)
IDESC_FORMAT = {"e4m3": 0, "e5m2": 1}


def product_table(fa: str, fb: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(neg, exp, mag, finite)`` of every (a, b) code pair, each (256, 256): the exact product as a Term (``exp`` the weight of
    bit 23, ``mag`` the PRODUCT_WIDTH-scaled significand product, 0 for a zero product) and whether both operands are finite."""

    FA, FB = FORMATS[fa], FORMATS[fb]
    neg = np.zeros((256, 256), bool)
    exp = np.zeros((256, 256), np.int64)
    mag = np.zeros((256, 256), np.int64)
    fin = np.zeros((256, 256), bool)
    dec_a = [FA.decode(w) for w in range(256)]
    dec_b = [FB.decode(w) for w in range(256)]
    shift = PRODUCT_WIDTH - (FA.sig_bits + FB.sig_bits)
    for a in range(256):
        da = dec_a[a]
        if isinstance(da, str):
            continue
        for b in range(256):
            db = dec_b[b]
            if isinstance(db, str):
                continue
            fin[a, b] = True
            (sa, ea, ma), (sb, eb, mb) = da, db
            if ma and mb:
                neg[a, b] = bool(sa ^ sb)
                exp[a, b] = ea + eb - shift + 23
                mag[a, b] = (ma * mb) << shift
    return neg, exp, mag, fin


_TABLES: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}


def tables(fa: str, fb: str) -> tuple[np.ndarray, ...]:
    key = (fa, fb)
    if key not in _TABLES:
        _TABLES[key] = product_table(fa, fb)
    return _TABLES[key]


def fp32_terms(words: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """FP32 words -> ``(neg, exp, mag, finite)`` as ``term.fp32_term`` decodes them (subnormals at exponent -126)."""

    w = np.asarray(words, dtype=np.uint32).astype(np.int64)
    e = (w >> 23) & 0xFF
    m = w & 0x7FFFFF
    neg = (w >> 31) == 1
    exp = np.where(e == 0, FP32_MIN_NORMAL_EXPONENT, e - 127)
    mag = np.where(e == 0, m, m | (1 << 23))
    return neg, exp, mag, e != 0xFF


def _rshift(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    return x >> np.clip(s, 0, 63)


def group_sum(neg: np.ndarray, exp: np.ndarray, mag: np.ndarray, *, width: int, zero_exponent: int):
    """``models.group_sum`` over the last axis of (n, t) term arrays: returns the (n,) result terms ``(neg, exp, mag)``."""

    live = mag != 0
    maxe = np.maximum(np.max(np.where(live, exp, _NEG_INF_EXP), axis=-1), zero_exponent)
    rescale = width - 24
    scaled = mag << rescale if rescale >= 0 else mag >> -rescale
    aligned = np.where(live, _rshift(scaled, maxe[..., None] - exp), 0)
    total = np.sum(np.where(neg, -aligned, aligned), axis=-1)
    rneg = total < 0
    rmag = np.abs(total)
    bl = np.frexp(rmag.astype(np.float64))[1].astype(np.int64)
    rexp = maxe + bl - width
    rmag = np.where(bl > width, _rshift(rmag, bl - width), rmag << np.clip(width - bl, 0, 63))
    sub = rexp < FP32_MIN_NORMAL_EXPONENT
    rmag = np.where(sub, _rshift(rmag, FP32_MIN_NORMAL_EXPONENT - rexp), rmag)
    rexp = np.where(sub, FP32_MIN_NORMAL_EXPONENT, rexp)
    rmag = rmag >> rescale if rescale >= 0 else rmag << -rescale
    zero = rmag == 0
    return np.where(zero, False, rneg), np.where(zero, 0, rexp), np.where(zero, 0, rmag)


def pack(neg: np.ndarray, exp: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """``term.pack_fp32`` on arrays: zero is +0, an exponent above 127 is the infinity of its sign."""

    field = exp + 127 - ((mag & (1 << 23)) == 0)
    w = (neg.astype(np.int64) << 31) | (np.clip(field, 0, 255) << 23) | (mag & 0x7FFFFF)
    w = np.where(exp > FP32_MAX_EXPONENT, (neg.astype(np.int64) << 31) | 0x7F800000, w)
    return np.where(mag == 0, 0, w).astype(np.uint32)


@dataclass(frozen=True)
class Candidate:
    """One hypothesis for a K = 32 FP8 step: ``GroupSum(groups, width, zero_exponent)`` over ``fa`` x ``fb`` products; a UE8M0
    scale pair, when given, adds ``(sa - 127) + (sb - 127)`` to the exponent of every product of the step before alignment."""

    name: str
    fa: str
    fb: str
    groups: tuple[int, ...] = (32,)
    width: int = 26
    zero_exponent: int = -133

    def step(self, c: np.ndarray, a: np.ndarray, b: np.ndarray, sa: np.ndarray | None = None, sb: np.ndarray | None = None):
        """One step on arrays: ``c`` (n,) FP32 words, ``a``/``b`` (n, 32) codes, ``sa``/``sb`` (n,) UE8M0 codes or None.
        Returns ``(words (n,) uint32, in_domain (n,) bool)``: outside the domain (a non-finite operand or accumulator, or a NaN
        scale) the word is meaningless."""

        neg_t, exp_t, mag_t, fin_t = tables(self.fa, self.fb)
        ai, bi = np.asarray(a, np.int64), np.asarray(b, np.int64)
        if ai.shape[-1] != sum(self.groups):
            raise ValueError(f"{self.name}: {ai.shape[-1]} products for groups {self.groups}")
        pn, pe, pm = neg_t[ai, bi], exp_t[ai, bi], mag_t[ai, bi]
        dom = fin_t[ai, bi].all(axis=-1)
        cn, ce, cm, cf = fp32_terms(c)
        dom &= cf
        if sa is not None:
            sa64, sb64 = np.asarray(sa, np.int64), np.asarray(sb, np.int64)
            dom &= (sa64 != 0xFF) & (sb64 != 0xFF)
            pe = pe + (sa64 + sb64 - 254)[:, None]
        inf = np.zeros(cn.shape, bool)
        inf_word = np.zeros(cn.shape, np.uint32)
        start = 0
        for size in self.groups:
            tn = np.concatenate([cn[:, None], pn[:, start:start + size]], axis=1)
            te = np.concatenate([ce[:, None], pe[:, start:start + size]], axis=1)
            tm = np.concatenate([cm[:, None], pm[:, start:start + size]], axis=1)
            cn, ce, cm = group_sum(tn, te, tm, width=self.width, zero_exponent=self.zero_exponent)
            over = (cm != 0) & (ce > FP32_MAX_EXPONENT) & ~inf
            inf_word = np.where(over, (cn.astype(np.uint32) << 31) | np.uint32(0x7F800000), inf_word)
            inf |= over
            ce = np.where(inf, 0, ce)
            start += size
        return np.where(inf, inf_word, pack(cn, ce, cm)), dom


def chain(cand: Candidate, c, a, b, sa=None, sb=None, steps: int = 1):
    """``steps`` steps of ``cand`` folded over K = 32 * steps codes (``a``/``b`` (n, 32 steps), ``sa``/``sb`` (n, steps))."""

    dom = np.ones(c.shape, bool)
    for s in range(steps):
        sl = slice(32 * s, 32 * (s + 1))
        c, d = cand.step(c, a[:, sl], b[:, sl], None if sa is None else sa[:, s], None if sb is None else sb[:, s])
        dom &= d
    return c, dom


def candidates(fa: str, fb: str, scaled: bool) -> dict[str, Candidate]:
    """The hypotheses a capture is scored against, sm_120's QMMA step first."""

    out: dict[str, Candidate] = {}

    def add(name, **kw):
        out[name] = Candidate(name, fa, fb, **kw)

    add("sm120_gs32_w26_f133")
    for w in (24, 25, 27, 28, 30):
        add(f"gs32_w{w}_f133", width=w)
    add("gs16x2_w26_f133", groups=(16, 16))
    add("gs8x4_w26_f133", groups=(8, 8, 8, 8))
    add("hopper_gs32_w14_f139", width=14, zero_exponent=-139)
    add("ada_gs16x2_w14_f139", groups=(16, 16), width=14, zero_exponent=-139)
    # Below 2^-126 only a UE8M0 scale can put every term of a step: the plain kinds' products are all above 2^-33.
    floors = (-120, -124, -126, -128, -130, -131, -132, -134, -135, -136, -140, -145, -149, -152, -155, -160, -174) if scaled else (-126, -149)
    for f in floors:
        add(f"gs32_w26_f{-f}", zero_exponent=f)
    return out
