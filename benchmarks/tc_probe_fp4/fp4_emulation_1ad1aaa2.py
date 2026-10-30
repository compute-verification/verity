"""Which non-native paths reproduce Pearl-C4's sm_120 NVFP4 chain words bit for bit (CPU, on the pinned models).

The work unit is ``NVFP4_SM120``'s atom (``verity.ml.tc.models.BLACKWELL_SM120_NVF4``): exact E2M1 x E2M1 group dots over
16, UE4M3 block scales, one five-way align-add truncating toward zero, RZ to binary32, chained over k/64 atoms.  A route
reproduces a word when its binary32 output equals the atom chain's.  Routes:

  native      the atom chain, vectorised here; ``selftest`` checks it word for word against the pinned model
  exact-rz    the exact block-scaled dot product, rounded once to binary32 toward zero: what any exact-integer pipeline
              produces when it accumulates in wide integers and converts at the end (INT8 IMMA group dots, an INT-only NPU,
              an ASIC with an exact accumulator)
  exact-rn    the same, rounded to nearest even
  bf16-fold   each E2M1 code times its UE4M3 block scale is exact in BF16 (6 significant bits), so the operands fold into
              BF16 and chain through sm_120's BF16 k16 step (``HOPPER_BF16_M16N8K16``, ``tc-model/sm120-bf16-k16``)
  step-floor  (hot accumulator only) H plus each atom's exact sum, truncated and floored onto the accumulator's grid

``--hot-bits h`` also runs every chain from a large initial accumulator H (the hot-accumulator tweak of the FP4
specialization study): the atom's RZ then floors each atom's contribution to 2^-h of a typical atom sum, so routes that
compute the exact sum stop matching.  Operands follow Pearl-C4's clean path with row-relative noise (``fp4-design.md``
section 1-2; i.i.d. noise stands in for the rank-r lines, as in ``fp4_census.py``).

    python benchmarks/pouw/fp4_emulation.py selftest
    python benchmarks/pouw/fp4_emulation.py routes --k 8192 --rows 64 --cols 64 --hot-bits 8,10,12 --out routes.json
    python benchmarks/pouw/fp4_emulation.py routes --hot-rule row --hot-bits 14,15,16 --out routes-row.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages" / "verity" / "src"))

from verity.ml.tc.models import BLACKWELL_SM120_NVF4, HOPPER_BF16_M16N8K16  # noqa: E402

#: E2M1 codes 0..7 as integer halves (sign in bit 3).
E2M1_HALVES = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.int64)
#: RN-even thresholds between consecutive E2M1 magnitudes: (midpoint, a tie rounds up).
_E2M1_MIDS = ((0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False))
NEG_INF = -(1 << 40)


# --- formats ---------------------------------------------------------------------------------------------------------

def e2m1_rn(v: np.ndarray) -> np.ndarray:
    """``cvt.rn.satfinite.e2m1``: nearest E2M1 code, ties to even, saturating at 6."""

    mag = np.abs(v)
    code = np.zeros(v.shape, np.int64)
    for mid, up in _E2M1_MIDS:
        code += (mag >= mid) if up else (mag > mid)
    return np.where((v < 0) & (code > 0), code | 8, code)


def ue4m3_rn(v: np.ndarray) -> np.ndarray:
    """``cvt.rn.satfinite`` of positive values to UE4M3 (bias 7, 3 mantissa bits, max 448, subnormals to 2^-9)."""

    v = np.asarray(v, np.float64)
    out = np.zeros(v.shape, np.int64)
    sub = v < 2.0 ** -6
    m_sub = np.rint(v / 2.0 ** -9).astype(np.int64)
    out = np.where(sub, np.where(m_sub >= 8, 1 << 3, m_sub), out)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.floor(np.log2(np.where(sub, 1.0, v))).astype(np.int64)
        m = np.rint((np.where(sub, 1.0, v) / np.exp2(e) - 1.0) * 8).astype(np.int64)
    e = np.where(m == 8, e + 1, e)
    m = np.where(m == 8, 0, m)
    ef = e + 7
    sat = (ef > 15) | ((ef == 15) & (m > 6))
    ef, m = np.where(sat, 15, ef), np.where(sat, 6, m)
    return np.where(sub, out, (ef << 3) | m)


def ue4m3_dyadic(code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """UE4M3 codes as ``(mantissa, exponent)`` with value ``mantissa * 2^exponent`` (the model's ``_scale_dyadic``)."""

    e, m = (code >> 3) & 0xF, code & 7
    return np.where(e == 0, m, 8 + m), np.where(e == 0, -9, e - 10)


def floor_log2(x: np.ndarray) -> np.ndarray:
    """``floor(log2 x)`` for positive int64 x, exact (float64 frexp can round a 54+-bit value up to a power of two)."""

    bl = np.frexp(x.astype(np.float64))[1].astype(np.int64)
    bl = np.where((x >> np.clip(bl - 1, 0, 62)) == 0, bl - 1, bl)
    return bl - 1


# --- Pearl-C4 operands -----------------------------------------------------------------------------------------------

def family(name: str, side: str, n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Real-valued rows (A: activations) or columns (B: weights) of one census family."""

    if name == "t4":
        return rng.standard_t(4, (n, k))
    x = rng.standard_normal((n, k))
    if name in ("massive", "massive3000") and side == "A":
        ch = np.random.default_rng(7).choice(k, 4, replace=False)   # the same channels in every row (massive activations)
        x[:, ch] *= 300.0 if name == "massive" else 3000.0
    if (name == "spread" and side == "A") or name == "spread-both":   # block scales spread over 12 binades (out of domain)
        x *= np.repeat(np.exp2(rng.uniform(-12, 0, (n, k // 16))), 16, axis=1)
    return x


def quantize_pearl_c4(x: np.ndarray, delta: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Pearl-C4's noisy NVFP4 operand: alpha0 from the row max and RMS, row-relative noise, noisy block scales, RN codes."""

    s = np.abs(x).max(axis=1, keepdims=True)
    rho = np.sqrt((x[:, ::8] ** 2).mean(axis=1, keepdims=True))
    sigma = delta * rho
    alpha0 = 2688.0 / np.maximum(s + 4 * sigma, 2.0 ** -32)
    v = alpha0 * (x + sigma * rng.standard_normal(x.shape))
    n, k = v.shape
    amax = np.abs(v).reshape(n, k // 16, 16).max(axis=2)
    sc = np.maximum(ue4m3_rn(amax / 6.0), 1)                          # floor at 2^-9 (code 1)
    ma, ea = ue4m3_dyadic(sc)
    scale_val = ma * np.exp2(ea)
    codes = e2m1_rn(v / np.repeat(scale_val, 16, axis=1))
    return codes, sc


def halves(codes: np.ndarray) -> np.ndarray:
    """Signed E2M1 values as integer halves."""

    h = E2M1_HALVES[codes & 7]
    return np.where(codes & 8, -h, h)


# --- the pinned atom, vectorised over words --------------------------------------------------------------------------

def trunc0(mant: np.ndarray, exp: np.ndarray, lsb: np.ndarray) -> np.ndarray:
    """``mant * 2^exp`` truncated toward zero onto the grid ``2^lsb``, as a multiple of ``2^lsb``."""

    d = exp - lsb
    left = mant << np.clip(d, 0, 62)
    right = np.abs(mant) >> np.clip(-d, 0, 62)
    right = np.where(-d > 62, 0, right)
    return np.where(d >= 0, left, np.where(mant < 0, -right, right))


def f32_rz(n: np.ndarray, lsb: np.ndarray) -> np.ndarray:
    """RZ binary32 bits of ``n * 2^lsb`` (the model's ``_to_f32_rz``); zero is +0, underflow to zero is +0."""

    neg = n < 0
    mag = np.abs(n)
    nz = mag != 0
    e_val = floor_log2(np.where(nz, mag, 1)) + lsb
    grid = np.where(e_val >= -126, e_val - 23, -149)
    shift = grid - lsb
    q = np.where(shift > 0, mag >> np.clip(shift, 0, 62), mag << np.clip(-shift, 0, 62))
    q = np.where(shift > 62, 0, q)
    e = floor_log2(np.where(q > 0, q, 1)) + grid
    if np.any(nz & (q > 0) & (e > 127)):
        raise ValueError("sum leaves binary32 (outside the model)")
    bits = np.where(e >= -126, ((e + 127) << 23) | (q & 0x7FFFFF), q)
    bits = np.where(neg, bits | (1 << 31), bits)
    return np.where(nz & (q > 0), bits, 0).astype(np.uint32)


def f32_rn(n: np.ndarray, lsb: np.ndarray) -> np.ndarray:
    """RN-even binary32 bits of ``n * 2^lsb`` (normal range only: the exact routes' final conversion)."""

    neg = n < 0
    mag = np.abs(n)
    nz = mag != 0
    shift = floor_log2(np.where(nz, mag, 1)) - 23
    sh = np.clip(shift, 0, 62)
    q = np.where(shift > 0, mag >> sh, mag << np.clip(-shift, 0, 62))
    rem = mag - (q << sh)
    half = np.where(shift > 0, np.int64(1) << np.clip(sh - 1, 0, 62), 0)
    up = (shift > 0) & ((rem > half) | ((rem == half) & (q & 1 == 1)))
    q = q + up
    carry = q >> 24
    q, shift = np.where(carry > 0, q >> 1, q), np.where(carry > 0, shift + 1, shift)
    e = shift + lsb + 23
    bits = ((e + 127) << 23) | (q & 0x7FFFFF)
    bits = np.where(neg, bits | (1 << 31), bits)
    return np.where(nz, bits, 0).astype(np.uint32)


def f32_dyadic(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """binary32 bits as ``(signed mantissa, exponent, exponent field)``."""

    c = c.astype(np.int64)
    ef = (c >> 23) & 0xFF
    frac = c & 0x7FFFFF
    m = np.where(ef == 0, frac, frac | (1 << 23))
    return np.where((c >> 31) & 1, -m, m), np.where(ef == 0, -149, ef - 150), ef


def nvf4_step(c: np.ndarray, mants: list, xs: list, parts: list) -> np.ndarray:
    """One ``mxf4nvf4`` 4X firing per word: the four scaled group dots ``mant * 2^X`` and the accumulator ``c``."""

    acc_m, acc_e, ef = f32_dyadic(c)
    anchor = np.full(c.shape, NEG_INF, np.int64)
    for x, p in zip(xs, parts):
        anchor = np.maximum(anchor, np.where(p, x - 27, NEG_INF))
    anchor = np.maximum(anchor, np.where(acc_m != 0, np.where(ef == 0, 1, ef) - 127 - 35, NEG_INF))
    has = anchor > NEG_INF
    lsb = np.maximum(anchor, -174)
    total = trunc0(acc_m, acc_e, lsb)
    for m, x in zip(mants, xs):
        total = total + trunc0(m, x, lsb)
    return np.where(has, f32_rz(total, lsb), 0).astype(np.uint32)


def atoms(ca: np.ndarray, sa: np.ndarray, cb: np.ndarray, sb: np.ndarray):
    """Per-block scaled group dots of every word: ``mant * 2^x`` and participation, each [K/16, R, C]."""

    rows, k = ca.shape
    cols, nblk = cb.shape[0], k // 16
    ha, hb = halves(ca), halves(cb)
    maa, eaa = ue4m3_dyadic(sa)
    mab, eab = ue4m3_dyadic(sb)
    a3 = ha.reshape(rows, nblk, 16).transpose(1, 0, 2).astype(np.float64)
    b3 = hb.reshape(cols, nblk, 16).transpose(1, 2, 0).astype(np.float64)
    gd = np.rint(a3 @ b3).astype(np.int64)                                      # exact: |gd| <= 2304
    nzp = np.rint(np.abs(a3) @ np.abs(b3)).astype(np.int64) > 0
    mm = maa.T[:, :, None] * mab.T[:, None, :]
    return gd, gd * mm, eaa.T[:, :, None] + eab.T[:, None, :] - 2, nzp & (mm != 0)


def native_chain(mant: np.ndarray, x: np.ndarray, part: np.ndarray, c0: np.ndarray, record: bool = False):
    """The atom chain over every word from ``c0``; with ``record``, also the number of (word, atom) identities."""

    c = c0.copy()
    same = 0
    for t in range(mant.shape[0] // 4):
        b = slice(4 * t, 4 * t + 4)
        nxt = nvf4_step(c, list(mant[b]), list(x[b]), list(part[b]))
        same += int((nxt == c).sum())
        c = nxt
    return (c, same) if record else c


class Problem:
    """R x C words of one family: codes and scales of A (rows) and B (columns), and the per-block group dots."""

    def __init__(self, fam: str, rows: int, cols: int, k: int, delta: float, seed: int):
        rng = np.random.default_rng(seed)
        ca, sa = quantize_pearl_c4(family(fam, "A", rows, k, rng), delta, rng)
        cb, sb = quantize_pearl_c4(family(fam, "B", cols, k, rng), delta, rng)
        self._set(ca, sa, cb, sb)

    @classmethod
    def from_codes(cls, ca: np.ndarray, sa: np.ndarray, cb: np.ndarray, sb: np.ndarray) -> "Problem":
        p = cls.__new__(cls)
        p._set(ca, sa, cb, sb)
        return p

    def _set(self, ca, sa, cb, sb) -> None:
        self.ca, self.sa, self.cb, self.sb = ca, sa, cb, sb
        self.k = ca.shape[1]
        self.nblk = self.k // 16
        self.ha, self.hb = halves(self.ca), halves(self.cb)
        self.maa, self.eaa = ue4m3_dyadic(self.sa)
        self.mab, self.eab = ue4m3_dyadic(self.sb)
        self.gd, self.mant, self.x, self.part = atoms(self.ca, self.sa, self.cb, self.sb)

    def native(self, c0: np.ndarray, record: bool = False):
        return native_chain(self.mant, self.x, self.part, c0, record)

    def step_sums(self) -> np.ndarray:
        """Each atom's exact sum as an integer at 2^-20 (the lowest NVFP4 unit weight): [steps, R, C]."""

        s = self.mant << (self.x + 20)
        return s.reshape(self.nblk // 4, 4, *s.shape[1:]).sum(axis=1)

    def exact(self, hot: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        s = self.step_sums().sum(axis=0)
        lsb = np.full(s.shape, -20, np.int64)
        if hot is not None:
            hm, he, _ = f32_dyadic(hot)
            s = s + (hm << (he + 20))
        return f32_rz(s, lsb), f32_rn(s, lsb)

    def step_floor(self, hot: np.ndarray) -> np.ndarray:
        """H + sum over atoms of floor_G(sum_g trunc0_L(term_g)), G = 2^(E_H - 23), L = 2^(E_H - 35)."""

        hm, he, ef = f32_dyadic(hot)
        eh = ef - 127
        total = hm << (he - (eh - 35))                                           # H on the grid L
        for t in range(self.nblk // 4):
            st = np.zeros(hot.shape, np.int64)
            for g in range(4 * t, 4 * t + 4):
                st = st + trunc0(self.mant[g], self.x[g], eh - 35)
            total = total + ((st >> 12) << 12)                                   # floor onto G = 2^12 L (H > 0)
        return f32_rz(total, eh - 35)

    def bf16_fold(self, c0: np.ndarray, blocks: range | None = None) -> np.ndarray:
        """The chain through sm_120's BF16 k16 step on the folded operands (E2M1 value x UE4M3 scale), from ``c0``."""

        fa = floor_log2(np.maximum(np.abs(self.ha) * self.maa.repeat(16, axis=1), 1))   # floor(log2 |a_num * ma|)
        fb = floor_log2(np.maximum(np.abs(self.hb) * self.mab.repeat(16, axis=1), 1))
        c = c0.copy()
        for blk in blocks if blocks is not None else range(self.nblk):
            ks = slice(16 * blk, 16 * blk + 16)
            ha, hb = self.ha[:, ks], self.hb[:, ks]
            mm = self.maa[:, blk][:, None] * self.mab[:, blk][None, :]
            e = self.eaa[:, blk][:, None] + self.eab[:, blk][None, :] - 2          # every product of the block is M * 2^e
            acc_m, acc_e, ef = f32_dyadic(c)
            maxe = np.where(acc_m != 0, np.where(ef == 0, -126, ef - 127), -133)
            for i in range(16):
                nz = (ha[:, i][:, None] != 0) & (hb[:, i][None, :] != 0)
                n_i = fa[:, 16 * blk + i][:, None] + fb[:, 16 * blk + i][None, :] + e
                maxe = np.maximum(maxe, np.where(nz, n_i, -133))
            g = maxe - 25
            total = trunc0(acc_m, acc_e, g)
            coarse = e < g
            if np.any(coarse):                                                   # products truncated one by one
                for i in range(16):
                    m_i = ha[:, i][:, None] * hb[:, i][None, :] * mm
                    total = total + np.where(coarse, trunc0(m_i, e, g), 0)
            total = total + np.where(coarse, 0, trunc0(self.mant[blk], e, g))    # on the grid, the block's sum is exact
            c = bf16_normalise(total, maxe)
        return c

    def bf16_floor(self, hot: np.ndarray) -> np.ndarray:
        """H plus each atom's BF16 k16 chain from +0 (4 steps), floored onto ulp(H): the GPU's ``bf16_floor`` route."""

        eh = ((hot.astype(np.int64) >> 23) & 0xFF) - 127
        units = np.full(hot.shape, 3 << 22, np.int64)                             # H in units of G = ulp(H) = 2^(E_H - 23)
        zero = np.zeros(hot.shape, np.uint32)
        for t in range(self.nblk // 4):
            st = self.bf16_fold(zero, range(4 * t, 4 * t + 4)).view(np.float32).astype(np.float64)
            units = units + np.floor(np.ldexp(st, -(eh - 23).astype(np.int32))).astype(np.int64)
        if np.any((units < (1 << 23)) | (units >= (1 << 24))):
            raise ValueError("bf16_floor left H's binade")
        return (((eh + 127) << 23) | (units - (1 << 23))).astype(np.uint32)


def bf16_normalise(total: np.ndarray, maxe: np.ndarray) -> np.ndarray:
    """``group_sum``'s tail at width 26: normalise ``total * 2^(maxe - 25)`` by truncation, then 24 bits (``pack_fp32``)."""

    neg = total < 0
    mag = np.abs(total)
    nz = mag != 0
    bl = floor_log2(np.where(nz, mag, 1)) + 1
    exponent = maxe + bl - 26
    mag = np.where(bl > 26, mag >> np.clip(bl - 26, 0, 62), mag << np.clip(26 - bl, 0, 62))
    sub = exponent < -126
    mag = np.where(sub, mag >> np.clip(-126 - exponent, 0, 62), mag)
    exponent = np.where(sub, -126, exponent)
    m24 = mag >> 2
    field = exponent + 127 - np.where(m24 & (1 << 23), 0, 1)
    if np.any(nz & (m24 > 0) & (exponent > 127)):
        raise ValueError("BF16 chain leaves binary32")
    bits = (field << 23) | (m24 & 0x7FFFFF)
    bits = np.where(neg, bits | (1 << 31), bits)
    return np.where(nz & (m24 > 0), bits, 0).astype(np.uint32)


def hot_accumulator(p: Problem, hot_bits: int, rule: str = "word") -> np.ndarray:
    """H = 1.5 * 2^E_H, with the grid 2^(E_H - 23) at 2^-hot_bits of the word's RMS atom sum (``word``), or of the row's
    largest one, one H per row (``row``: the cheap form, H held in the accumulator registers)."""

    s = p.step_sums().astype(np.float64) * 2.0 ** -20
    rms = np.sqrt((s ** 2).mean(axis=0))
    if rule == "row":
        rms = np.broadcast_to(rms.max(axis=1, keepdims=True), rms.shape)
    eh = np.floor(np.log2(np.maximum(rms, 2.0 ** -100))).astype(np.int64) + 23 - hot_bits
    return ((eh + 127) << 23 | (1 << 22)).astype(np.uint32)


def merged_k128(p: Problem, c0: np.ndarray) -> np.ndarray:
    """The chain with each pair of atoms merged into one align-add over 8 groups: the 2:4-sparse m16n8k128 route on a 2:4
    span, which merges two atoms' align-adds (GPU 4's measured merged-vs-chained differences rule out two internal k64 steps;
    the one-add-over-8-groups form is this model's hypothesis)."""

    c = c0.copy()
    for t in range(p.nblk // 8):
        b = slice(8 * t, 8 * t + 8)
        c = nvf4_step(c, list(p.mant[b]), list(p.x[b]), list(p.part[b]))
    return c


# --- commands --------------------------------------------------------------------------------------------------------

def _bf16_bits(half_num: int, mant: int, exp: int) -> int:
    """BF16 bits of ``half_num/2 * mant * 2^exp`` (exact: at most 6 significant bits)."""

    v = abs(half_num) * mant
    if v == 0:
        return 0
    fl = v.bit_length() - 1
    frac = (v << (7 - fl)) & 0x7F if fl <= 7 else (v >> (fl - 7)) & 0x7F
    field = fl + exp - 1 + 127
    return ((1 << 15) if half_num < 0 else 0) | (field << 7) | frac


def selftest(args) -> dict:
    """The vectorised atom and BF16 step against the pinned models, word for word, on every family (hot and cold)."""

    out = {}
    for fam in ("gaussian", "massive", "spread-both"):
        p = Problem(fam, 6, 5, 1024, 0.25, args.seed)
        for label, c0 in (("cold", np.zeros((6, 5), np.uint32)), ("hot", hot_accumulator(p, 10))):
            nat, bf = p.native(c0), p.bf16_fold(c0)
            bad_n = bad_b = 0
            for r in range(6):
                for c in range(5):
                    want = BLACKWELL_SM120_NVF4.chain(int(c0[r, c]), [int(v) for v in p.ca[r]], [int(v) for v in p.cb[c]],
                                                      [int(v) for v in p.sa[r]], [int(v) for v in p.sb[c]])
                    bad_n += int(want != int(nat[r, c]))
                    acc = int(c0[r, c])
                    for blk in range(p.nblk):
                        a_w = [_bf16_bits(int(p.ha[r, 16 * blk + i]), int(p.maa[r, blk]), int(p.eaa[r, blk])) for i in range(16)]
                        b_w = [_bf16_bits(int(p.hb[c, 16 * blk + i]), int(p.mab[c, blk]), int(p.eab[c, blk])) for i in range(16)]
                        acc = HOPPER_BF16_M16N8K16.step(acc, a_w, b_w)
                    bad_b += int(acc != int(bf[r, c]))
            rz, _ = p.exact(c0 if label == "hot" else None)
            out[f"{fam}/{label}"] = {"words": 30, "native_mismatches": bad_n, "bf16_fold_mismatches": bad_b,
                                     "native_ne_exact": int((rz != nat).sum()), "native_ne_bf16": int((bf != nat).sum())}
    ok = all(v["native_mismatches"] == 0 and v["bf16_fold_mismatches"] == 0 for v in out.values())
    # the check must see truncation: words where the atom differs from the exact sum and from the BF16 chain
    out["ok"] = ok and sum(v["native_ne_exact"] for v in out.values()) > 0 and sum(v["native_ne_bf16"] for v in out.values()) > 0
    return out


def routes(args) -> dict:
    fams = args.families.split(",")
    hot_bits = [int(h) for h in args.hot_bits.split(",") if h]
    res = {"k": args.k, "rows": args.rows, "cols": args.cols, "delta": args.delta, "seed": args.seed,
           "atom": BLACKWELL_SM120_NVF4.name, "bf16_step": HOPPER_BF16_M16N8K16.name, "families": {}}
    for fam in fams:
        t0 = time.time()
        p = Problem(fam, args.rows, args.cols, args.k, args.delta, args.seed)
        zero = np.zeros((args.rows, args.cols), np.uint32)
        nat = p.native(zero)
        rz, rn = p.exact()
        bf = p.bf16_fold(zero)
        n = nat.size
        cold = {"words": n, "exact_rz": float((rz == nat).mean()), "exact_rn": float((rn == nat).mean()),
                "bf16_fold": float((bf == nat).mean()), "merged_k128": float((merged_k128(p, zero) == nat).mean())}
        hot = {}
        s_exact = p.step_sums().sum(axis=0).astype(np.float64) * 2.0 ** -20
        for h in hot_bits:
            H = hot_accumulator(p, h, args.hot_rule)
            eh = ((H.astype(np.int64) >> 23) & 0xFF) - 127
            nat_h, same = p.native(H, record=True)
            rz_h, _ = p.exact(H)
            sf = p.step_floor(H)
            bf_h = p.bf16_fold(H)
            hv = H.view(np.float32).astype(np.float64)
            err = (nat_h.view(np.float32).astype(np.float64) - hv) - s_exact
            scale = np.sqrt((s_exact ** 2).mean())
            hot[str(h)] = {"identity_atoms": same / (n * (p.nblk // 4)), "exact_rz": float((rz_h == nat_h).mean()),
                           "step_floor": float((sf == nat_h).mean()), "bf16_fold": float((bf_h == nat_h).mean()),
                           "merged_k128": float((merged_k128(p, H) == nat_h).mean()),
                           # (word, block) pairs whose largest possible term (2^19 at 2^X) is below H's inner grid 2^(E_H - 35):
                           # they add exactly 0, so the scales alone say they can be skipped
                           "dead_pairs": float(((p.x + 19) < (eh - 35)[None]).mean()),
                           "rel_err_rms": float(np.sqrt((err ** 2).mean()) / scale),
                           "rel_bias": float(err.mean() / scale)}
        res["families"][fam] = {"cold": cold, "hot": hot, "seconds": round(time.time() - t0, 1)}
        print(fam, json.dumps(res["families"][fam]), flush=True)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "routes"])
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--cols", type=int, default=64)
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--families", default="gaussian,t4,massive,spread,spread-both")
    ap.add_argument("--hot-bits", default="8,10,12")
    ap.add_argument("--hot-rule", choices=["word", "row"], default="word")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = selftest(args) if args.cmd == "selftest" else routes(args)
    text = json.dumps(res, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text if args.cmd == "selftest" else f"wrote {args.out}")
    return 0 if res.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
