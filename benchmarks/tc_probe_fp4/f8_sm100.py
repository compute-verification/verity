"""Capture and score the B200's (sm_100a) ``tcgen05.mma`` FP8 step, ``kind::f8f6f4`` and ``kind::mxf8f6f4``.

Three phases, so that only the capture holds a GPU (``f8_sm100.sh`` runs them in order)::

    python f8_sm100.py gen     --scratch DIR [--seed S] [--tiles N] [--bulk-scale F]   # CPU: every family's operands, .npy
    python f8_sm100.py capture --scratch DIR --lib libtcgen05_f8.so --out OUT          # GPU: layout gate, then every family
    python f8_sm100.py eval    --scratch DIR --out OUT [--procs P]                      # CPU: every output against the models

Kinds (``KINDS``): ``kind::f8f6f4`` with E4M3 x E4M3, E5M2 x E5M2 and both mixed pairs; ``kind::mxf8f6f4
.block_scale.scale_vec::1X`` with UE8M0 scales, E4M3 x E4M3, E5M2 x E5M2 and E4M3 x E5M2.  A tile is one CTA: M = 128 rows,
N = 32 columns, K = 32 x steps (``steps`` chained MMAs on one TMEM accumulator), so 4,096 outputs a tile.

Families (``FAMILIES``; each kind runs every family that applies): ``random_*`` and ``uniform_bits`` (random codes and
accumulators), ``acc_walk`` / ``two_product_walk`` (one or two products against an accumulator walked across every candidate's
alignment edge), ``spread`` (the largest-exponent spread: one top product and the rest 0..40 binades below), ``cancellation``
(exact cancellation of k 0..15 against 16..31 with residuals, and tiny survivors against cancelling large products: Hawkeye's
adder-width probe), ``subnormal_operands``, ``acc_only`` (zero products, every FP32 accumulator class), ``tiny_acc_products``,
``signed_zero``, ``enable0`` (enable_input_d = 0 with a garbage accumulator), ``chain2/4/8`` (K = 64..256 on one accumulator,
random and binade-growing), ``specials`` (record-only: NaN/inf operands and accumulators), and for mxf8f6f4 ``scale_sweep``,
``scale_extremes`` (products pushed to 2^-254: the floor), ``scale_overflow`` and ``scale_random``.  ``bulk`` is the random set
(``--bulk-scale`` 1.0 = 10^8 outputs over the kinds), sampled from its own seed after the adversarial families.

``eval`` scores every in-domain output against ``f8_model.candidates`` (sm_120's QMMA step ``GroupSum (32,)/26/-133`` first) and
writes ``summary.json`` (per kind and family: outputs, asserted outputs, mismatches per candidate, observed -0 / subnormal / inf
words, first mismatches as hex records) and ``fixture.json`` (captured records sampled per family, the pinned vectors).
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools" / "config"))
import f8_model as FM  # noqa: E402
from verity_config.settings import SHARED  # noqa: E402

M_T, N_T, K_STEP = 128, 32, 32
SEED = 20261019
BULK_SEED = 20261020

#: kind -> (tcgen05 kernel kind: 0 kind::f8f6f4, 1 kind::mxf8f6f4; A format; B format)
KINDS = {
    "f8_e4m3": (0, "e4m3", "e4m3"),
    "f8_e5m2": (0, "e5m2", "e5m2"),
    "f8_e4m3_e5m2": (0, "e4m3", "e5m2"),
    "f8_e5m2_e4m3": (0, "e5m2", "e4m3"),
    "mx_e4m3": (1, "e4m3", "e4m3"),
    "mx_e5m2": (1, "e5m2", "e5m2"),
    "mx_e4m3_e5m2": (1, "e4m3", "e5m2"),
}
#: bulk outputs per kind at --bulk-scale 1 (10^8 in all)
BULK = {"f8_e4m3": 30_000_000, "f8_e5m2": 20_000_000, "f8_e4m3_e5m2": 10_000_000, "f8_e5m2_e4m3": 5_000_000,
        "mx_e4m3": 20_000_000, "mx_e5m2": 10_000_000, "mx_e4m3_e5m2": 5_000_000}
RECORD_ONLY = {"specials"}

# --- formats ---------------------------------------------------------------------------------------------------------------


def _fmt_info(fmt: str):
    F = FM.FORMATS[fmt]
    vals = np.full(256, np.nan)
    finite = np.zeros(256, bool)
    for w in range(256):
        d = F.decode(w)
        if isinstance(d, str):
            continue
        s, e, m = d
        vals[w] = (-1.0 if s else 1.0) * m * 2.0 ** e
        finite[w] = True
    return F, vals, finite


INFO = {f: _fmt_info(f) for f in ("e4m3", "e5m2")}


def finite_codes(fmt: str) -> np.ndarray:
    return np.flatnonzero(INFO[fmt][2]).astype(np.uint8)


def codes(rng, fmt: str, shape, nonzero: bool = False) -> np.ndarray:
    pool = finite_codes(fmt)
    if nonzero:
        pool = pool[(pool & 0x7F) != 0]
    return pool[rng.integers(0, len(pool), shape)]


def code_of(fmt: str, sign, e_field, mant) -> np.ndarray:
    F = FM.FORMATS[fmt]
    c = (np.asarray(sign, np.int64) << 7) | (np.asarray(e_field, np.int64) << F.man_bits) | np.asarray(mant, np.int64)
    return c.astype(np.uint8)


def value(fmt: str, c) -> np.ndarray:
    return INFO[fmt][1][np.asarray(c, np.int64)]


def max_e_field(fmt: str) -> int:
    return 15 if fmt == "e4m3" else 30


def f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def acc_near(rng, p: np.ndarray, d_lo: int, d_hi: int) -> np.ndarray:
    """FP32 words +-(1.f) 2^(floor(log2|p|) + d), d uniform in [d_lo, d_hi], mantissa 0 / all ones / random in thirds."""
    lp = np.floor(np.log2(np.where(p == 0, 1.0, np.abs(p)))).astype(np.int64)
    e = np.clip(lp + rng.integers(d_lo, d_hi + 1, p.shape) + 127, 1, 254).astype(np.uint32)
    kind = rng.integers(0, 3, p.shape)
    f = np.where(kind == 0, 0, np.where(kind == 1, (1 << 23) - 1, rng.integers(0, 1 << 23, p.shape))).astype(np.uint32)
    s = rng.integers(0, 2, p.shape).astype(np.uint32)
    return (s << 31) | (e << 23) | f


def uniform_acc(rng, shape, e_lo: int = 0, e_hi: int = 254) -> np.ndarray:
    e = rng.integers(e_lo, e_hi + 1, shape).astype(np.uint32)
    m = rng.integers(0, 1 << 23, shape).astype(np.uint32)
    s = rng.integers(0, 2, shape).astype(np.uint32)
    return (s << 31) | (e << 23) | m


def acc_random(rng, shape, lo=-30, hi=40) -> np.ndarray:
    return f32(rng.standard_normal(shape) * np.exp2(rng.integers(lo, hi, shape)))


# --- families ---------------------------------------------------------------------------------------------------------------


class Tiles(dict):
    """A family's arrays: A (n,128,32S) u8, B (n,32,32S) u8, SFA (n,128,S) u8, SFB (n,32,S) u8, C (n,128,32) u32, EN (n,) u8."""


def blank(n: int, steps: int = 1, scaled: bool = False, rng=None) -> Tiles:
    K = K_STEP * steps
    t = Tiles(A=np.zeros((n, M_T, K), np.uint8), B=np.zeros((n, N_T, K), np.uint8),
              SFA=np.full((n, M_T, steps), 127, np.uint8), SFB=np.full((n, N_T, steps), 127, np.uint8),
              C=np.zeros((n, M_T, N_T), np.uint32), EN=np.ones(n, np.uint8), steps=steps)
    if scaled and rng is not None:
        t["SFA"] = rng.integers(117, 138, (n, M_T, steps)).astype(np.uint8)
        t["SFB"] = rng.integers(117, 138, (n, N_T, steps)).astype(np.uint8)
    return t


def scale_val(t: Tiles, s: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return np.exp2(t["SFA"][:, :, s].astype(np.float64) - 127), np.exp2(t["SFB"][:, :, s].astype(np.float64) - 127)


def prod_max(t: Tiles, fa: str, fb: str, s: int = 0) -> np.ndarray:
    """(n,128,32): the largest |product| of each output's step s (scales included)."""
    sl = slice(K_STEP * s, K_STEP * (s + 1))
    pa = np.abs(np.nan_to_num(value(fa, t["A"][:, :, sl])))
    pb = np.abs(np.nan_to_num(value(fb, t["B"][:, :, sl])))
    pm = np.einsum("nik,njk->nijk", pa, pb).max(-1) if pa.shape[0] * M_T * N_T * K_STEP < 3e8 else _pmax_loop(pa, pb)
    sa, sb = scale_val(t, s)
    return pm * sa[:, :, None] * sb[:, None, :]


def _pmax_loop(pa, pb):
    out = np.zeros((pa.shape[0], M_T, N_T))
    for i in range(0, pa.shape[0], 64):
        out[i:i + 64] = np.einsum("nik,njk->nijk", pa[i:i + 64], pb[i:i + 64]).max(-1)
    return out


def fam_random_zero_acc(rng, n, fa, fb, scaled):
    t = blank(n, scaled=scaled, rng=rng)
    t["A"], t["B"] = codes(rng, fa, t["A"].shape), codes(rng, fb, t["B"].shape)
    t["C"][n // 2:] = 0x80000000
    return t


def fam_random_random_acc(rng, n, fa, fb, scaled):
    t = blank(n, scaled=scaled, rng=rng)
    t["A"], t["B"] = codes(rng, fa, t["A"].shape), codes(rng, fb, t["B"].shape)
    t["C"] = acc_near(rng, prod_max(t, fa, fb), -30, 30)
    return t


def fam_uniform_bits(rng, n, fa, fb, scaled):
    t = blank(n, scaled=scaled, rng=rng)
    t["A"], t["B"] = codes(rng, fa, t["A"].shape), codes(rng, fb, t["B"].shape)
    if scaled:
        t["SFA"] = rng.integers(0, 255, t["SFA"].shape).astype(np.uint8)
        t["SFB"] = np.minimum(rng.integers(0, 255, t["SFB"].shape), 254).astype(np.uint8)
        # keep sa + sb <= 254 + 100 so most sums stay finite; the overflow family covers the rest
        cap = (354 - t["SFA"].max(axis=1, keepdims=True).astype(np.int64)).clip(0, 254)
        t["SFB"] = np.minimum(t["SFB"].astype(np.int64), cap).astype(np.uint8)
    t["C"] = uniform_acc(rng, t["C"].shape)
    return t


def _one_hot(rng, t, fa, fb, nprod):
    n = t["A"].shape[0]
    ks = np.stack([rng.permutation(K_STEP)[:nprod] for _ in range(n * M_T)]).reshape(n, M_T, nprod)
    for j in range(nprod):
        np.put_along_axis(t["A"], ks[:, :, j:j + 1], codes(rng, fa, (n, M_T, 1), nonzero=True), axis=2)
    t["B"] = codes(rng, fb, t["B"].shape, nonzero=True)


def fam_acc_walk(rng, n, fa, fb, scaled):
    t = blank(n, scaled=scaled, rng=rng)
    _one_hot(rng, t, fa, fb, 1)
    t["C"] = acc_near(rng, prod_max(t, fa, fb), -45, 30)
    return t


def fam_two_product_walk(rng, n, fa, fb, scaled):
    t = blank(n, scaled=scaled, rng=rng)
    _one_hot(rng, t, fa, fb, 2)
    t["C"] = acc_near(rng, prod_max(t, fa, fb), -40, 10)
    t["C"][: n // 4] = 0
    return t


def fam_spread(rng, n, fa, fb, scaled):
    """Largest-exponent spread: each row's top code at k0, the others 0..(range) binades below it; B random nonzero."""
    t = blank(n, scaled=scaled, rng=rng)
    F = FM.FORMATS[fa]
    emax = max_e_field(fa) - (1 if fa == "e5m2" else 0)
    top = rng.integers(max(1, emax - 4), emax + 1, (n, M_T, 1))
    drop = rng.integers(0, emax + 1, (n, M_T, K_STEP))
    e = np.clip(top - drop, 0, emax)
    m = rng.integers(0, 1 << F.man_bits, (n, M_T, K_STEP))
    m = np.where((fa == "e4m3") & (e == 15) & (m == 7), 6, m)
    k0 = rng.integers(0, K_STEP, (n, M_T, 1))
    e = np.where(np.arange(K_STEP)[None, None, :] == k0, top, e)
    t["A"] = code_of(fa, rng.integers(0, 2, e.shape), e, m)
    t["B"] = codes(rng, fb, t["B"].shape, nonzero=True)
    third = n // 3
    t["C"][third:2 * third] = acc_near(rng, prod_max({**t, "A": t["A"][third:2 * third], "B": t["B"][third:2 * third],
                                                      "SFA": t["SFA"][third:2 * third], "SFB": t["SFB"][third:2 * third]}, fa, fb),
                                       -5, 5)
    t["C"][2 * third:] = acc_random(rng, t["C"][2 * third:].shape, -40, 10)
    return t


def fam_cancellation(rng, n, fa, fb, scaled):
    t = blank(n, scaled=scaled, rng=rng)
    A = codes(rng, fa, (n, M_T, 16), nonzero=True)
    B = codes(rng, fb, (n, N_T, 16), nonzero=True)
    t["A"] = np.concatenate([A, A], axis=2)
    t["B"] = np.concatenate([B, B ^ 0x80], axis=2)
    h = n // 2
    # first half: a few residual products at random k in the second half (A replaced)
    res = rng.random((h, M_T, 16)) < 0.1
    t["A"][:h, :, 16:] = np.where(res, codes(rng, fa, (h, M_T, 16)), t["A"][:h, :, 16:])
    # second half: one pair zeroed and a tiny code put in its first slot -> a tiny survivor against cancelling large products
    j0 = rng.integers(0, 16, (n - h, M_T, 1))
    big_e = max_e_field(fa) - (2 if fa == "e5m2" else 1)
    t["A"][h:, :, :16] = code_of(fa, rng.integers(0, 2, (n - h, M_T, 16)), rng.integers(big_e - 3, big_e + 1, (n - h, M_T, 16)),
                                 rng.integers(0, 1 << FM.FORMATS[fa].man_bits, (n - h, M_T, 16)))
    t["A"][h:, :, 16:] = t["A"][h:, :, :16]
    tiny = code_of(fa, rng.integers(0, 2, (n - h, M_T, 1)), rng.integers(0, big_e, (n - h, M_T, 1)),
                   rng.integers(0, 1 << FM.FORMATS[fa].man_bits, (n - h, M_T, 1)))
    np.put_along_axis(t["A"][h:], j0, tiny, axis=2)
    np.put_along_axis(t["A"][h:], j0 + 16, np.zeros_like(tiny), axis=2)
    q = n // 4
    t["C"][q:h] = acc_random(rng, t["C"][q:h].shape, -40, -10)
    t["C"][h + (n - h) // 2:] = 0x80000000
    return t


def fam_subnormal_operands(rng, n, fa, fb, scaled):
    t = blank(n, scaled=scaled, rng=rng)
    Fa, Fb = FM.FORMATS[fa], FM.FORMATS[fb]

    def low(fmt, F, shape):
        e = np.where(rng.random(shape) < 0.6, 0, rng.integers(1, 3, shape))
        c = code_of(fmt, rng.integers(0, 2, shape), e, rng.integers(0, 1 << F.man_bits, shape))
        mix = rng.random(shape) < 0.25
        return np.where(mix, codes(rng, fmt, shape), c)

    t["A"], t["B"] = low(fa, Fa, t["A"].shape), low(fb, Fb, t["B"].shape)
    third = n // 3
    t["C"][third:2 * third] = uniform_acc(rng, t["C"][third:2 * third].shape, 0, 0)
    t["C"][2 * third:] = acc_near(rng, np.maximum(prod_max(t, fa, fb)[2 * third:], 2.0 ** -40), -30, 10)
    return t


def fam_acc_only(rng, n, fa, fb, scaled):
    """Zero products (A codes +-0) against every FP32 accumulator class: subnormal, tiny normal, uniform, LSB patterns."""
    t = blank(n, scaled=scaled, rng=rng)
    t["A"] = np.where(rng.random(t["A"].shape) < 0.5, 0, 0x80).astype(np.uint8)
    t["B"] = codes(rng, fb, t["B"].shape)
    q = n // 4
    t["C"][:q] = uniform_acc(rng, t["C"][:q].shape, 0, 0)
    t["C"][q:2 * q] = uniform_acc(rng, t["C"][q:2 * q].shape, 1, 8)
    t["C"][2 * q:3 * q] = uniform_acc(rng, t["C"][2 * q:3 * q].shape)
    base = uniform_acc(rng, (n - 3 * q, M_T, 1))
    t["C"][3 * q:] = (base & ~np.uint32(0x3FF)) | rng.integers(0, 1024, (n - 3 * q, M_T, N_T)).astype(np.uint32)
    return t


def fam_tiny_acc_products(rng, n, fa, fb, scaled):
    t = blank(n, scaled=scaled, rng=rng)
    t["A"], t["B"] = codes(rng, fa, t["A"].shape), codes(rng, fb, t["B"].shape)
    sparse = rng.random(t["A"].shape) < 0.9
    t["A"] = np.where(sparse, 0, t["A"]).astype(np.uint8)
    t["C"] = uniform_acc(rng, t["C"].shape, 0, 30)
    return t


def fam_signed_zero(rng, n, fa, fb, scaled):
    t = blank(n, scaled=scaled, rng=rng)
    t["A"] = np.where(rng.random(t["A"].shape) < 0.5, 0, 0x80).astype(np.uint8)
    t["B"] = codes(rng, fb, t["B"].shape)
    h = n // 2
    one_a = 0x38 if fa == "e4m3" else 0x3C
    three_b = (0x44, 0xC4) if fb == "e4m3" else (0x42, 0xC2)
    t["A"][h:, :, :16] = one_a
    t["B"][h:, :, :16] = np.where(np.arange(16) % 2 == 0, three_b[0], three_b[1]).astype(np.uint8)
    t["B"][h:, :, 16:] = codes(rng, fb, t["B"][h:, :, 16:].shape)
    t["C"] = np.where(rng.random(t["C"].shape) < 0.5, np.uint32(0x80000000), np.uint32(0)).astype(np.uint32)
    t["EN"][n // 4: h] = 0
    t["EN"][h + (n - h) // 2:] = 0
    return t


def fam_enable0(rng, n, fa, fb, scaled):
    t = fam_random_random_acc(rng, n, fa, fb, scaled)
    t["C"] = uniform_acc(rng, t["C"].shape)
    t["EN"][:] = 0
    return t


def _chain(rng, n, fa, fb, scaled, steps):
    t = blank(n, steps, scaled=scaled, rng=rng)
    t["A"], t["B"] = codes(rng, fa, t["A"].shape), codes(rng, fb, t["B"].shape)
    h = n // 2
    # second half: every product positive and near one magnitude, so the accumulator climbs binades step after step
    if fa == "e4m3":
        t["A"][h:] = code_of(fa, 0, rng.integers(7, 9, t["A"][h:].shape), rng.integers(0, 8, t["A"][h:].shape))
    else:
        t["A"][h:] = code_of(fa, 0, rng.integers(15, 17, t["A"][h:].shape), rng.integers(0, 4, t["A"][h:].shape))
    t["B"][h:] = t["B"][h:] & 0x7F
    t["C"][: h // 2] = acc_near(rng, prod_max(t, fa, fb)[: h // 2], -20, 20)
    t["C"][h // 2: h] = uniform_acc(rng, t["C"][h // 2: h].shape, 100, 150)
    return t


def fam_chain2(rng, n, fa, fb, scaled):
    return _chain(rng, n, fa, fb, scaled, 2)


def fam_chain4(rng, n, fa, fb, scaled):
    return _chain(rng, n, fa, fb, scaled, 4)


def fam_chain8(rng, n, fa, fb, scaled):
    return _chain(rng, n, fa, fb, scaled, 8)


def fam_specials(rng, n, fa, fb, scaled):
    t = blank(n, scaled=scaled, rng=rng)
    t["A"], t["B"] = codes(rng, fa, t["A"].shape), codes(rng, fb, t["B"].shape)
    nan_a = (0x7F, 0xFF) if fa == "e4m3" else (0x7C, 0xFC, 0x7D, 0xFE)
    nan_b = (0x7F, 0xFF) if fb == "e4m3" else (0x7C, 0xFC, 0x7E, 0xFF)
    t["A"][:, :, 0] = np.where(rng.random((n, M_T)) < 0.3, np.array(nan_a, np.uint8)[rng.integers(0, len(nan_a), (n, M_T))],
                               t["A"][:, :, 0])
    t["B"][:, :, 1] = np.where(rng.random((n, N_T)) < 0.3, np.array(nan_b, np.uint8)[rng.integers(0, len(nan_b), (n, N_T))],
                               t["B"][:, :, 1])
    sp = np.array([0x7FC00000, 0xFFC00000, 0x7F800001, 0x7F800000, 0xFF800000, 0x7FFFFFFF], np.uint32)
    t["C"] = np.where(rng.random(t["C"].shape) < 0.5, sp[rng.integers(0, len(sp), t["C"].shape)], acc_random(rng, t["C"].shape))
    if scaled:
        t["SFA"][:, :8] = 0xFF
    return t


def fam_scale_sweep(rng, n, fa, fb, scaled):
    t = blank(n, scaled=True, rng=rng)
    t["A"], t["B"] = codes(rng, fa, t["A"].shape), codes(rng, fb, t["B"].shape)
    sa = ((np.arange(n)[:, None] * M_T + np.arange(M_T)[None, :]) % 255).astype(np.int64)
    sb = np.full((n, N_T), 127, np.int64)
    h = n // 2
    sb[h:] = (np.arange(n - h)[:, None] * N_T + np.arange(N_T)[None, :] + 97) % 255
    sb = np.minimum(sb, 330 - sa.max(axis=1, keepdims=True))
    t["SFA"], t["SFB"] = sa.astype(np.uint8)[:, :, None], sb.clip(0, 254).astype(np.uint8)[:, :, None]
    t["C"][h:] = acc_random(rng, t["C"][h:].shape)
    return t


def fam_scale_extremes(rng, n, fa, fb, scaled):
    """Products pushed toward 2^-254: sums below the subnormal grid, the adder's floor, subnormal accumulators."""
    t = blank(n, scaled=True, rng=rng)
    t["A"], t["B"] = codes(rng, fa, t["A"].shape), codes(rng, fb, t["B"].shape)
    lo = np.concatenate([np.arange(0, 16), np.arange(40, 64)])
    t["SFA"] = lo[rng.integers(0, len(lo), t["SFA"].shape)].astype(np.uint8)
    t["SFB"] = lo[rng.integers(0, len(lo), t["SFB"].shape)].astype(np.uint8)
    q = n // 4
    t["C"][q:2 * q] = uniform_acc(rng, t["C"][q:2 * q].shape, 0, 0)
    t["C"][2 * q:3 * q] = uniform_acc(rng, t["C"][2 * q:3 * q].shape, 1, 12)
    t["C"][3 * q:] = acc_near(rng, np.maximum(prod_max(t, fa, fb)[3 * q:], 2.0 ** -300), -20, 20)
    return t


def fam_scale_floor_walk(rng, n, fa, fb, scaled):
    """One product per output, its scale pair walked so the product's exponent sweeps -180..-110, against +0 / subnormal acc."""
    t = blank(n, scaled=True, rng=rng)
    _one_hot(rng, t, fa, fb, 1)
    pe = np.floor(np.log2(np.maximum(prod_max({**t, "SFA": np.full_like(t["SFA"], 127), "SFB": np.full_like(t["SFB"], 127)},
                                              fa, fb), 2.0 ** -60)))
    target = rng.integers(-180, -105, (n, M_T, N_T)).astype(np.float64)
    # one scale pair per (row, column): split the needed shift between SFA (per row) and SFB (per column)
    need = (target - pe)[:, :, 0]
    sa = np.clip(np.round(need / 2) + 127, 0, 254)
    t["SFA"] = sa.astype(np.uint8)[:, :, None]
    sbn = (target - pe - (sa[:, :, None] - 127))[:, 0, :]
    t["SFB"] = np.clip(np.round(sbn) + 127, 0, 254).astype(np.uint8)[:, :, None]
    h = n // 2
    t["C"][h:] = uniform_acc(rng, t["C"][h:].shape, 0, 0)
    return t


def pow2_code(fmt: str, p) -> np.ndarray:
    """The code of +2^p (subnormal codes below the normal range), p in [min subnormal, max power of two]."""
    F = FM.FORMATS[fmt]
    p = np.asarray(p, np.int64)
    e_field = p + F.bias
    sub = e_field < 1
    mant = np.where(sub, 1 << np.clip(F.man_bits + e_field - 1, 0, 7), 0)
    return code_of(fmt, 0, np.where(sub, 0, e_field), mant)


def pow2_range(fmt: str) -> tuple[int, int]:
    F = FM.FORMATS[fmt]
    return 1 - F.bias - F.man_bits, max_e_field(fmt) - F.bias


def fam_scale_floor_probe(rng, n, fa, fb, scaled):
    """The adder's floor: products only (acc +-0), n_i equal powers of two 2^m (m = p_i + s, the top exponent, swept
    -165..-110) against one tiny product of the opposite sign 2^(m - d).  The tiny one moves the RZ result down one 2^-149 step
    iff it is kept, which is iff m - d >= max(m, floor) - (width - 1): the floor shows where the threshold stops following m."""
    t = blank(n, scaled=True, rng=rng)
    lo_a, hi_a = pow2_range(fa)
    lo_b, _ = pow2_range(fb)
    p = rng.integers(max(lo_a, hi_a - 12), hi_a + 1, (n, M_T))
    qa = rng.integers(lo_a, hi_a + 1, (n, M_T))
    qb = rng.integers(lo_b, 3, (n, N_T))
    nbig = np.array([1, 2, 4, 8, 16, 30])[rng.integers(0, 6, (n, M_T))]
    neg_big = rng.random((n, M_T)) < 0.5
    tiny_sign = np.where(rng.random((n, M_T)) < 0.85, ~neg_big, neg_big)
    k = np.arange(K_STEP)[None, None, :]
    big = pow2_code(fa, p)[:, :, None] | (neg_big.astype(np.uint8) << 7)[:, :, None]
    tiny = pow2_code(fa, qa) | (tiny_sign.astype(np.uint8) << 7)
    tiny = np.where(rng.random((n, M_T)) < 0.08, 0, tiny).astype(np.uint8)
    t["A"] = np.where(k < nbig[:, :, None], big, 0).astype(np.uint8)
    t["A"][:, :, K_STEP - 1] = tiny
    one_b = pow2_code(fb, 0)
    t["B"] = np.broadcast_to(np.where(k < K_STEP - 1, one_b, 0), (n, N_T, K_STEP)).astype(np.uint8)
    t["B"][:, :, K_STEP - 1] = pow2_code(fb, qb)
    X = rng.integers(-90, -49, (n, M_T))
    Y = rng.integers(-75, -59, (n, N_T))
    t["SFA"] = np.clip(127 + X - p, 1, 254).astype(np.uint8)[:, :, None]
    t["SFB"] = np.clip(127 + Y, 1, 254).astype(np.uint8)[:, :, None]
    t["C"] = np.where(rng.random(t["C"].shape) < 0.5, np.uint32(0), np.uint32(0x80000000)).astype(np.uint32)
    return t


def fam_scale_overflow(rng, n, fa, fb, scaled):
    t = blank(n, scaled=True, rng=rng)
    t["A"], t["B"] = codes(rng, fa, t["A"].shape), codes(rng, fb, t["B"].shape)
    t["SFA"] = rng.integers(200, 255, t["SFA"].shape).astype(np.uint8)
    t["SFB"] = rng.integers(150, 255, t["SFB"].shape).astype(np.uint8)
    h = n // 2
    t["C"][h:] = uniform_acc(rng, t["C"][h:].shape, 240, 254)
    return t


def fam_scale_random(rng, n, fa, fb, scaled):
    t = blank(n, scaled=True, rng=rng)
    t["A"], t["B"] = codes(rng, fa, t["A"].shape), codes(rng, fb, t["B"].shape)
    t["SFA"] = rng.integers(60, 200, t["SFA"].shape).astype(np.uint8)
    t["SFB"] = rng.integers(60, 200, t["SFB"].shape).astype(np.uint8)
    t["C"] = acc_near(rng, np.maximum(prod_max(t, fa, fb), 2.0 ** -200), -30, 30)
    return t


FAMILIES = {
    "random_zero_acc": (fam_random_zero_acc, 1, False),
    "random_random_acc": (fam_random_random_acc, 1, False),
    "uniform_bits": (fam_uniform_bits, 1, False),
    "acc_walk": (fam_acc_walk, 1, False),
    "two_product_walk": (fam_two_product_walk, 1, False),
    "spread": (fam_spread, 1, False),
    "cancellation": (fam_cancellation, 1, False),
    "subnormal_operands": (fam_subnormal_operands, 1, False),
    "acc_only": (fam_acc_only, 1, False),
    "tiny_acc_products": (fam_tiny_acc_products, 1, False),
    "signed_zero": (fam_signed_zero, 1, False),
    "enable0": (fam_enable0, 0.5, False),
    "chain2": (fam_chain2, 0.5, False),
    "chain4": (fam_chain4, 0.5, False),
    "chain8": (fam_chain8, 0.5, False),
    "specials": (fam_specials, 0.25, False),
    "scale_sweep": (fam_scale_sweep, 1, True),
    "scale_extremes": (fam_scale_extremes, 1, True),
    "scale_floor_walk": (fam_scale_floor_walk, 1, True),
    "scale_floor_probe": (fam_scale_floor_probe, 2, True),
    "scale_overflow": (fam_scale_overflow, 0.5, True),
    "scale_random": (fam_scale_random, 1, True),
}


def fam_bulk(rng, n, fa, fb, scaled):
    t = blank(n, scaled=scaled, rng=rng)
    t["A"], t["B"] = codes(rng, fa, t["A"].shape), codes(rng, fb, t["B"].shape)
    if scaled:
        t["SFA"] = rng.integers(100, 155, t["SFA"].shape).astype(np.uint8)
        t["SFB"] = rng.integers(100, 155, t["SFB"].shape).astype(np.uint8)
    q = n // 4
    t["C"][:q] = 0
    t["C"][q:3 * q] = acc_near(rng, np.maximum(prod_max({k: (v[q:3 * q] if isinstance(v, np.ndarray) else v) for k, v in t.items()},
                                                        fa, fb), 2.0 ** -100), -30, 30)
    t["C"][3 * q:] = uniform_acc(rng, t["C"][3 * q:].shape)
    return t


# --- gen / capture / eval ------------------------------------------------------------------------------------------------------


ARRAYS = ("A", "B", "SFA", "SFB", "C", "EN")


def _path(scratch: Path, kind: str, fam: str) -> Path:
    return scratch / f"{kind}__{fam}"


def save_set(scratch: Path, kind: str, fam: str, t) -> None:
    d = _path(scratch, kind, fam)
    d.mkdir(parents=True, exist_ok=True)
    for k in ARRAYS:
        np.save(d / f"{k}.npy", np.ascontiguousarray(t[k]))
    (d / "steps").write_text(str(int(t["steps"])))


def load_set(scratch: Path, kind: str, fam: str) -> dict:
    """A family's arrays, memory-mapped, and its ``steps``; ``D`` too once captured."""
    d = _path(Path(scratch), kind, fam)
    z = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ARRAYS}
    z["steps"] = int((d / "steps").read_text())
    if (d / "D.npy").exists():
        z["D"] = np.load(d / "D.npy", mmap_mode="r")
    return z


def gen(args) -> int:
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    plan = []
    for ki, (kind, (kk, fa, fb)) in enumerate(KINDS.items()):
        scaled = kk == 1
        for fi, (fam, (fn, mult, needs_scale)) in enumerate(FAMILIES.items()):
            if needs_scale and not scaled:
                continue
            n = max(2, int(round(args.tiles * mult)))
            rng = np.random.default_rng([args.seed, ki, fi])
            t = fn(rng, n, fa, fb, scaled)
            save_set(scratch, kind, fam, t)
            plan.append({"kind": kind, "family": fam, "tiles": n, "steps": t["steps"], "outputs": n * M_T * N_T})
        nb = int(round(BULK[kind] * args.bulk_scale / (M_T * N_T)))
        if nb:
            parts = []
            for p in range(0, nb, 512):
                rng = np.random.default_rng([BULK_SEED, ki, p])
                parts.append(fam_bulk(rng, min(512, nb - p), fa, fb, scaled))
            t = Tiles({k: (np.concatenate([q[k] for q in parts]) if isinstance(parts[0][k], np.ndarray) else parts[0][k]) for k in parts[0]})
            save_set(scratch, kind, "bulk", t)
            plan.append({"kind": kind, "family": "bulk", "tiles": nb, "steps": 1, "outputs": nb * M_T * N_T})
    (scratch / "plan.json").write_text(json.dumps({"seed": args.seed, "bulk_seed": BULK_SEED, "families": plan}, indent=1))
    print(f"gen: {len(plan)} sets, {sum(p['outputs'] for p in plan):,} outputs", flush=True)
    return 0


class Lib:
    def __init__(self, path: str):
        self.lib = ctypes.CDLL(path)
        f = self.lib.tcgen05_f8_tiles
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_int, ctypes.c_uint] + [ctypes.c_void_p] * 7 + [ctypes.c_int, ctypes.c_int]
        self.lib.tcgen05_f8_idesc.restype = ctypes.c_uint
        self.lib.tcgen05_f8_idesc.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_uint]

    def identity(self) -> dict:
        major, minor = ctypes.c_int(), ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        rc = self.lib.device_identity(ctypes.byref(major), ctypes.byref(minor), name, 256)
        return {"rc": rc, "cc": f"{major.value}.{minor.value}", "name": name.value.decode()}

    def run(self, kind: str, t, batch: int = 2048) -> np.ndarray:
        kk, fa, fb = KINDS[kind]
        idesc = self.lib.tcgen05_f8_idesc(kk, FM.IDESC_FORMAT[fa], FM.IDESC_FORMAT[fb])
        n, steps = t["A"].shape[0], int(t["steps"])
        D = np.empty((n, M_T, N_T), np.uint32)
        for i in range(0, n, batch):
            j = min(n, i + batch)
            arrs = [np.ascontiguousarray(t[k][i:j]) for k in ("A", "B", "SFA", "SFB", "C", "EN")]
            out = np.empty((j - i, M_T, N_T), np.uint32)
            rc = self.lib.tcgen05_f8_tiles(kk, idesc, *[a.ctypes.data for a in arrs], out.ctypes.data, j - i, steps)
            if rc:
                raise RuntimeError(f"CUDA error {rc} on {kind}")
            D[i:j] = out
        return D


def _small_ints(rng, fmt, shape):
    vals = {"e4m3": {0: 0, 1: 0x38, 2: 0x40, 3: 0x44, 4: 0x48, 6: 0x4C, 8: 0x50},
            "e5m2": {0: 0, 1: 0x3C, 2: 0x40, 3: 0x42, 4: 0x44, 6: 0x46, 8: 0x48}}[fmt]
    keys = np.array(sorted(vals))
    v = keys[rng.integers(0, len(keys), shape)] * np.where(rng.random(shape) < 0.5, -1, 1)
    lut = np.zeros(17, np.uint8)
    for k, c in vals.items():
        lut[k] = c
    c = lut[np.abs(v)] | np.where(v < 0, 0x80, 0).astype(np.uint8)
    return v.astype(np.float64), c.astype(np.uint8)


def layout_gate(lib: Lib, rng) -> dict:
    """Small exact integers (and for mxf8f6f4 scales 2^-2..2^2): every output must equal the exact A.B + C, for each kind at
    steps 1, 2 and 8.  This catches a misrouted row, column, K chunk, step or scale before any capture is believed."""
    out = {}
    for kind, (kk, fa, fb) in KINDS.items():
        for steps in (1, 2, 8):
            n = 8
            t = blank(n, steps)
            va, t["A"] = _small_ints(rng, fa, t["A"].shape)
            vb, t["B"] = _small_ints(rng, fb, t["B"].shape)
            if kk == 1:
                t["SFA"] = rng.integers(125, 130, t["SFA"].shape).astype(np.uint8)
                t["SFB"] = rng.integers(125, 130, t["SFB"].shape).astype(np.uint8)
            cv = rng.integers(-64, 65, t["C"].shape).astype(np.float64)
            t["C"] = f32(cv)
            t["EN"][n // 2:] = 0
            exact = np.zeros((n, M_T, N_T))
            for s in range(steps):
                sl = slice(K_STEP * s, K_STEP * (s + 1))
                sa = np.exp2(t["SFA"][:, :, s].astype(np.float64) - 127)
                sb = np.exp2(t["SFB"][:, :, s].astype(np.float64) - 127)
                exact += np.einsum("nik,njk->nij", va[:, :, sl] * sa[:, :, None], vb[:, :, sl] * sb[:, :, None])
            exact += cv * t["EN"][:, None, None]
            D = lib.run(kind, t)
            bad = int((D != f32(exact)).sum())
            out[f"{kind}/steps{steps}"] = {"outputs": int(D.size), "mismatches": bad, "unwritten": int((D == 0xEEEEEEEE).sum())}
    return out


def capture(args, lib=None) -> int:
    scratch, out = Path(args.scratch), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lib = lib or Lib(args.lib)
    ident = lib.identity()
    gate = layout_gate(lib, np.random.default_rng(7))
    kind_ok = {kind: all(v["mismatches"] == 0 and v["unwritten"] == 0 for k, v in gate.items() if k.startswith(kind + "/"))
               for kind in KINDS}
    ok = all(kind_ok.values())
    doc = {"identity": ident, "host": platform.node(), "layout_gate": gate, "layout_ok": ok, "kind_layout_ok": kind_ok,
           "captured": []}
    print(json.dumps({"identity": ident, "layout_ok": ok, "kind_layout_ok": kind_ok}), flush=True)
    if not ok:
        print(json.dumps(gate, indent=1), flush=True)
    plan = json.loads((scratch / "plan.json").read_text())
    for p in plan["families"]:
        if not kind_ok[p["kind"]]:
            continue
        z = load_set(scratch, p["kind"], p["family"])
        D = lib.run(p["kind"], z)
        np.save(_path(scratch, p["kind"], p["family"]) / "D.npy", D)
        doc["captured"].append({**p, "unwritten": int((D == 0xEEEEEEEE).sum()),
                                "sha256_D": hashlib.sha256(D.tobytes()).hexdigest()})
        print(f"captured {p['kind']}/{p['family']}: {D.size:,}", flush=True)
    (out / "capture.json").write_text(json.dumps(doc, indent=1))
    return 0 if ok else 3


# --- eval ---------------------------------------------------------------------------------------------------------------


def expand(z: dict, i: int, j: int):
    """Tiles i..j -> per-output arrays: c (n,), a (n, K), b (n, K), sa (n, S), sb (n, S), en (n,)."""
    A, B = z["A"][i:j], z["B"][i:j]
    nt, K = A.shape[0], A.shape[2]
    a = np.broadcast_to(A[:, :, None, :], (nt, M_T, N_T, K)).reshape(-1, K)
    b = np.broadcast_to(B[:, None, :, :], (nt, M_T, N_T, K)).reshape(-1, K)
    S = z["SFA"].shape[2]
    sa = np.broadcast_to(z["SFA"][i:j][:, :, None, :], (nt, M_T, N_T, S)).reshape(-1, S)
    sb = np.broadcast_to(z["SFB"][i:j][:, None, :, :], (nt, M_T, N_T, S)).reshape(-1, S)
    en = np.repeat(z["EN"][i:j], M_T * N_T)
    c = np.where(en == 1, z["C"][i:j].reshape(-1), 0).astype(np.uint32)
    return c, a, b, sa, sb, en


def _eval_job(job):
    scratch, kind, fam, i, j, names = job
    kk, fa, fb = KINDS[kind]
    z = load_set(scratch, kind, fam)
    D = np.asarray(z["D"][i:j]).reshape(-1)
    c, a, b, sa, sb, en = expand(z, i, j)
    steps = z["steps"]
    cands = FM.candidates(fa, fb, kk == 1)
    res = {"outputs": int(D.size), "unwritten": int((D == 0xEEEEEEEE).sum()), "neg_zero": int((D == 0x80000000).sum()),
           "subnormal": int((((D >> 23) & 0xFF) == 0).sum() - (D == 0).sum() - (D == 0x80000000).sum()),
           "inf": int(((D & 0x7FFFFFFF) == 0x7F800000).sum()), "nan": int((((D >> 23) & 0xFF) == 0xFF).sum() - ((D & 0x7FFFFFFF) == 0x7F800000).sum()),
           "models": {}, "examples": {}}
    dom = None
    for name in names:
        cand = cands[name]
        w, d = FM.chain(cand, c, a, b, sa if kk == 1 else None, sb if kk == 1 else None, steps)
        if dom is None:
            dom = d
            res["asserted"] = int(d.sum())
        bad = d & (w != D)
        res["models"][name] = int(bad.sum())
        idx = np.flatnonzero(bad)[:6]
        if len(idx):
            res["examples"][name] = [{"c": f"{int(c[k]):08X}", "a": bytes(a[k]).hex(), "b": bytes(b[k]).hex(),
                                      "sa": bytes(sa[k]).hex(), "sb": bytes(sb[k]).hex(), "d": f"{int(D[k]):08X}",
                                      "model": f"{int(w[k]):08X}", "en": int(en[k])} for k in idx]
    return kind, fam, res


def _merge(acc, res):
    if not acc:
        return res
    for k in ("outputs", "unwritten", "neg_zero", "subnormal", "inf", "nan", "asserted"):
        acc[k] += res[k]
    for k, v in res["models"].items():
        acc["models"][k] += v
    for k, v in res["examples"].items():
        acc["examples"].setdefault(k, [])
        acc["examples"][k] = (acc["examples"][k] + v)[:12]
    return acc


def fixture_records(scratch: Path, kind: str, fam: str, rng, per: int) -> list[dict]:
    z = load_set(scratch, kind, fam)
    D = z["D"]
    nt = z["A"].shape[0]
    recs = []
    for _ in range(per):
        ti, m, n = int(rng.integers(0, nt)), int(rng.integers(0, M_T)), int(rng.integers(0, N_T))
        c, a, b, sa, sb, en = expand(z, ti, ti + 1)
        k = m * N_T + n
        recs.append({"kind": kind, "family": fam, "steps": z["steps"], "c": f"{int(c[k]):08X}", "a": bytes(a[k]).hex(),
                     "b": bytes(b[k]).hex(), "sa": bytes(sa[k]).hex() if KINDS[kind][0] == 1 else "",
                     "sb": bytes(sb[k]).hex() if KINDS[kind][0] == 1 else "", "d": f"{int(D[ti, m, n]):08X}"})
    return recs


def evaluate(args) -> int:
    scratch, out = Path(args.scratch), Path(args.out)
    plan = json.loads((scratch / "plan.json").read_text())
    plan["families"] = [p for p in plan["families"] if (_path(scratch, p["kind"], p["family"]) / "D.npy").exists()]
    jobs = []
    for p in plan["families"]:
        kk, fa, fb = KINDS[p["kind"]]
        names = list(FM.candidates(fa, fb, kk == 1))
        if p["family"] == "bulk" and not args.bulk_all_models:
            names = names[:1] + [x for x in args.bulk_models.split(",") if x and x in names and x != names[0]]
        chunk = max(1, (args.chunk_outputs // (M_T * N_T)) // max(1, p["steps"]))
        for i in range(0, p["tiles"], chunk):
            jobs.append((str(scratch), p["kind"], p["family"], i, min(p["tiles"], i + chunk), names))
    summary: dict = {}
    with ProcessPoolExecutor(args.procs) as ex:
        for kind, fam, res in ex.map(_eval_job, jobs):
            summary.setdefault(kind, {})
            summary[kind][fam] = _merge(summary[kind].get(fam), res)
    totals = {}
    for kind, fams in summary.items():
        for fam, r in fams.items():
            asserted = fam not in RECORD_ONLY
            for name, bad in r["models"].items():
                tt = totals.setdefault(kind, {}).setdefault(name, {"asserted": 0, "mismatches": 0, "adversarial_asserted": 0,
                                                                   "adversarial_mismatches": 0, "bulk_asserted": 0, "bulk_mismatches": 0})
                if not asserted:
                    continue
                tt["asserted"] += r["asserted"]
                tt["mismatches"] += bad
                part = "bulk" if fam == "bulk" else "adversarial"
                tt[f"{part}_asserted"] += r["asserted"]
                tt[f"{part}_mismatches"] += bad
    doc = {"plan": plan, "totals": totals, "families": summary}
    (out / "summary.json").write_text(json.dumps(doc, indent=1))
    rng = np.random.default_rng(99)
    recs = []
    for p in plan["families"]:
        recs += fixture_records(scratch, p["kind"], p["family"], rng, 4 if p["family"] == "bulk" else 24)
    (out / "fixture.json").write_text(json.dumps({"device": "B200 (sm_100a)", "records": recs}, indent=0))
    for kind, t in totals.items():
        best = sorted(t.items(), key=lambda kv: kv[1]["mismatches"])[:4]
        print(kind, " | ".join(f"{n}: {v['mismatches']:,}/{v['asserted']:,}" for n, v in best), flush=True)
    return 0


# --- pin: the repository's fixtures from one run's out/ ----------------------------------------------------------------------

PIN_SELECTION = (
    "Selection (deterministic): for each kind, from every asserted family of the run (``specials`` is record-only and left out), "
    "the families in plan order and in each the candidates in f8_model.candidates order: the first device word of the "
    "family's recorded mismatches (summary.json ``examples``) that a candidate misses, unless a record already selected for "
    "that kind refutes it; then the first ``uniform`` records of the family's uniform sample (fixture.json).  Each record "
    "once.  Every selected word equals sm120_gs32_w26_f133's (the run's 0-mismatch candidate); pin asserts it.  ``c`` is the "
    "accumulator the first step reads (+0 on enable_input_d = 0 tiles); ``a``/``b`` are 32 codes a step, ``sa``/``sb`` one "
    "UE8M0 byte a step (kind::mxf8f6f4 only)."
)


def _codes(h: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(h), np.uint8)


def replay(rec: dict, cand: "FM.Candidate") -> tuple[int, bool]:
    """One fixture record under ``cand``: (word, in_domain)."""
    a, b = _codes(rec["a"])[None, :], _codes(rec["b"])[None, :]
    sa = _codes(rec["sa"])[None, :] if rec.get("sa") else None
    sb = _codes(rec["sb"])[None, :] if rec.get("sb") else None
    w, d = FM.chain(cand, np.array([int(rec["c"], 16)], np.uint32), a, b, sa, sb, a.shape[1] // K_STEP)
    return int(w[0]), bool(d[0])


def pin(args) -> int:
    src = Path(args.run_out)
    summary = json.loads((src / "summary.json").read_text())
    uniform_recs = json.loads((src / "fixture.json").read_text())["records"]
    capture_doc = json.loads((src / "capture.json").read_text())
    truth = "sm120_gs32_w26_f133"
    out: dict[str, list[dict]] = {}
    for kind, fams in summary["families"].items():
        kk, fa, fb = KINDS[kind]
        cands = FM.candidates(fa, fb, kk == 1)
        chosen: list[dict] = []
        seen: set[tuple] = set()

        def take(rec: dict, fam: str) -> None:
            r = {"family": fam, "c": rec["c"].upper(), "a": rec["a"], "b": rec["b"],
                 "sa": rec["sa"] if kk == 1 else "", "sb": rec["sb"] if kk == 1 else "", "d": rec["d"].upper()}
            key = (r["c"], r["a"], r["b"], r["sa"], r["sb"])
            if key in seen:
                return
            word, dom = replay(r, cands[truth])
            assert dom and word == int(r["d"], 16), (kind, r)
            seen.add(key)
            chosen.append(r)

        refuted: set[str] = set()
        for fam, res in fams.items():
            if fam in RECORD_ONLY:
                continue
            for name in cands:
                if name == truth or name in refuted or not res["examples"].get(name):
                    continue
                take(res["examples"][name][0], fam)
                refuted |= {n for n, c in cands.items() if replay(chosen[-1], c)[0] != int(chosen[-1]["d"], 16)}
            for rec in [r for r in uniform_recs if r["kind"] == kind and r["family"] == fam][: args.uniform]:
                take(rec, fam)
        out[kind] = chosen
    gpu = (src / "gpu.csv").read_text().strip().splitlines()[-1] if (src / "gpu.csv").exists() else ""
    nvcc = (src / "nvcc.txt").read_text().strip().splitlines()[-2:] if (src / "nvcc.txt").exists() else []
    prov = {"run": args.run, "device": f"{capture_doc['identity']['name']} (cc {capture_doc['identity']['cc']})",
            "nvidia_smi": gpu, "nvcc": " / ".join(nvcc), "host": capture_doc["host"], "seed": summary["plan"]["seed"], "bulk_seed": summary["plan"]["bulk_seed"],
            "totals": {k: v[truth] for k, v in summary["totals"].items()}, "selection": PIN_SELECTION}
    doc = {"schema": "verity/tc-capture-sample/v0", "provenance": {**prov, "uniform": args.uniform}, "kinds": out}
    Path(args.out).write_text(json.dumps(doc, indent=0) + "\n")
    for kind, recs in out.items():
        print(kind, len(recs), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("gen", "capture", "eval", "pin"))
    ap.add_argument("--scratch")
    ap.add_argument("--run-out", help="pin: a run's out/ (summary.json, fixture.json, capture.json)")
    ap.add_argument("--run", help="pin: that run's id")
    ap.add_argument("--uniform", type=int, default=2, help="pin: uniform records a family")
    ap.add_argument("--out", default=SHARED.OUT or ".")
    ap.add_argument("--lib")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tiles", type=int, default=24, help="tiles per adversarial family (x its multiplier); 4,096 outputs a tile")
    ap.add_argument("--bulk-scale", type=float, default=1.0, help="1.0 = 10^8 bulk outputs over the kinds")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--chunk-outputs", type=int, default=1 << 19)
    ap.add_argument("--bulk-models", default="", help="comma list of extra candidates scored on bulk (sm_120's always is)")
    ap.add_argument("--bulk-all-models", action="store_true")
    args = ap.parse_args(argv)
    if args.phase != "pin" and not args.scratch:
        ap.error(f"{args.phase} needs --scratch")
    return {"gen": gen, "capture": capture, "eval": evaluate, "pin": pin}[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
