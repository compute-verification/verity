"""Served sparse noise (SSN) on a floating-point tensor core: the formed operands, the accumulator init and the noise.

ssn_1l's layout, per proof block of ``block`` steps (``None``: the whole K is one block, as fast accumulation runs it):
the block's first step holds L - 4 data lanes and 4 code lanes, a[i, e] in {0, 1} against the column's public code
C[e, j] in [-6, 6]^4 \\ {0} (injective, base 13); every later step holds L - 1 data lanes and one noise lane a_t[i] in
{0, 1} against the public weight 1.  L is 32 (FP8 k32) or 16 (BF16 k16).  The noise lane of row i carries u_i, so a
noise product is u_i times a small integer.

Designs (``mode``):

* ``stock``: the data alone, accumulator 0 (no noise; every word is a salt-independent function of the input).
* ``fixed``: ssn_1l literally (u = 1, accumulator init -beta_j): the int8 design on the float pipeline.
* ``pin``: u_i = 2^(p_i - w) with w the word's fraction bits (13 for FP8, 23 for BF16) and the accumulator init the row's
  bias 1.5 * 2^p_i, p_i chosen so every partial sum of the block, every product and the noise stay inside half the
  binade [2^p_i, 2^(p_i + 1)): the accumulator always holds the largest exponent, the grid is u_i at every step and the
  sum never renormalises, so each word is exactly bias + sum of grid-truncated data products + u_i * (integer noise).
  ``bound`` picks p_i: ``cs`` from Cauchy-Schwarz (max_j |a_i| |b_j| over the block, needs nothing but norms) or
  ``partial`` from the data's own largest partial sum and product (a prover's committed choice that the drawn proof
  block's Program checks: every word's exponent is p_i).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from verity.ml.circuits.silicon.term import FORMATS

from . import sim

CODE_LANES = 4
CODE_BASE = 13


def code_table(n: int) -> np.ndarray:
    """C[e, j]: ssn_1l's injective base-13 code of column j, digits in [-6, 6], never all zero."""
    c = np.arange(1, n + 1)
    c = np.where(c < 14280, c, c + 1)
    if n > 28559:
        raise ValueError("the 4-lane base-13 code is injective for n <= 28,559")
    return np.stack([(c // CODE_BASE**e) % CODE_BASE - 6 for e in range(CODE_LANES)])


def _finite_values(dtype: str) -> tuple[np.ndarray, np.ndarray]:
    f = FORMATS[dtype]
    codes = np.arange(1 << (f.bits - 1))
    vals = []
    keep = []
    for c in codes:
        d = f.decode(int(c))
        if isinstance(d, str):
            continue
        _, e, m = d
        keep.append(c)
        vals.append(m * 2.0**e)
    order = np.argsort(vals)
    return np.asarray(keep)[order], np.asarray(vals)[order]


_TABLES: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def encode(dtype: str, x: np.ndarray) -> np.ndarray:
    """x to the nearest finite value of the format (saturating), as operand words."""
    if dtype == "bf16":
        b = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
        r = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
        return r.astype(np.uint16)
    if dtype not in _TABLES:
        _TABLES[dtype] = _finite_values(dtype)
    codes, vals = _TABLES[dtype]
    f = FORMATS[dtype]
    ax = np.minimum(np.abs(x), vals[-1])
    hi = np.clip(np.searchsorted(vals, ax), 1, len(vals) - 1)
    lo = hi - 1
    pick = np.where(ax - vals[lo] <= vals[hi] - ax, lo, hi)
    out = codes[pick].astype(np.int64)
    neg = (x < 0) & (vals[pick] > 0)
    return (out | np.where(neg, 1 << (f.bits - 1), 0)).astype(np.uint8)


def decode(dtype: str, w: np.ndarray) -> np.ndarray:
    if dtype == "bf16":
        return (w.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if dtype not in _TABLES:
        _TABLES[dtype] = _finite_values(dtype)
    codes, vals = _TABLES[dtype]
    f = FORMATS[dtype]
    lut = np.zeros(1 << f.bits)
    lut[codes] = vals
    lut[codes | (1 << (f.bits - 1))] = -vals
    return lut[w.astype(np.int64)]


@dataclass(frozen=True)
class Layout:
    """Where each lane of the formed operands comes from, for K data entries."""

    K: int
    L: int
    block: int | None  # steps a block; None: one block

    def build(self):
        data_lane, code_lane, noise_lane, step_block = [], [], [], []
        k = s = 0
        b = -1
        while k < self.K:
            first = self.block is None and s == 0 or self.block is not None and s % self.block == 0
            if first:
                b += 1
            base = s * self.L
            nd = self.L - (CODE_LANES if first else 1)
            take = min(nd, self.K - k)
            data_lane += [(base + q, k + q) for q in range(take)]
            if first:
                code_lane += [(base + self.L - CODE_LANES + e, b, e) for e in range(CODE_LANES)]
            else:
                noise_lane.append((base + self.L - 1, s))
            step_block.append(b)
            k += take
            s += 1
        return s, b + 1, data_lane, code_lane, noise_lane, step_block

    @property
    def steps(self) -> int:
        return self.build()[0]


def pipe_of(dtype: str) -> sim.Pipe:
    return {"e4m3": sim.E4M3_K32, "e5m2": sim.E5M2_K32, "bf16": sim.BF16_K16}[dtype]


def frac_bits(dtype: str) -> int:
    """Fraction bits of a word in the pinned binade: the adder's for FP8 (14 bits), FP32's for BF16 (the 26-bit sum is
    truncated back to 24 bits), so u = 2^(p - frac_bits) is the word's ulp there."""
    return min(pipe_of(dtype).width, 24) - 1


def pow2_word(dtype: str, r: np.ndarray) -> np.ndarray:
    """The operand word of 2^r (exact; r inside the format's range)."""
    return encode(dtype, np.exp2(r.astype(np.float64)))


#: the exact powers of two of each format (exponent range, subnormals included) and the weight side's public scale 2^sigma
#: on the code and noise lanes (the largest with 6 * 2^sigma in range), so a noise product is 2^(r + sigma) * integer
POW2 = {"e4m3": (-9, 8), "e5m2": (-16, 15), "bf16": (-133, 127)}
SIGMA = {"e4m3": 6, "e5m2": 13, "bf16": 0}


@dataclass
class Formed:
    A: np.ndarray  # (m, K') words
    B: np.ndarray  # (n, K') words
    init: np.ndarray  # (m, n, blocks) FP32 words: each block's accumulator init
    u: np.ndarray  # (m,) noise unit (value of one noise product unit)
    noise: np.ndarray  # (m, n, S) integer noise of each step's word, in units of u
    steps: int
    blocks: int
    step_block: np.ndarray  # (S,)
    bias: np.ndarray  # (m, n, blocks) init values (float64)
    epi: np.ndarray  # (m, n, blocks) what the epilogue subtracts from each block's last word (pin: 1.5 * 2^p_i)
    p: np.ndarray | None = None  # (m,) pinned binade exponents
    unpinnable: np.ndarray | None = None  # (m,) rows whose bound needs a unit beyond the format's powers of two


def form(dtype: str, A: np.ndarray, B: np.ndarray, mode: str, coins: np.random.Generator, block: int | None = None,
         bound: str = "partial", rmin: int | None = None, rmax: int | None = None) -> Formed:
    """The formed operands of one call: A (m, K) and B (n, K) data words, coins for this salt."""
    p_ = pipe_of(dtype)
    m, K = A.shape
    n = B.shape[0]
    lay = Layout(K, p_.k, block)
    S, nb, data_lane, code_lane, noise_lane, step_block = lay.build()
    Kp = S * p_.k
    word = np.uint8 if dtype != "bf16" else np.uint16
    Ap = np.zeros((m, Kp), word)
    Bp = np.zeros((n, Kp), word)
    dl = np.array([d[0] for d in data_lane])
    dk = np.array([d[1] for d in data_lane])
    Ap[:, dl] = A[:, dk]
    Bp[:, dl] = B[:, dk]
    step_block = np.asarray(step_block)
    if mode == "stock":
        init = np.zeros((m, n, nb))
        return Formed(Ap, Bp, init.astype(np.float32).view(np.uint32), np.zeros(m), np.zeros((m, n, S), np.int64), S, nb,
                      step_block, init, init)
    C = code_table(n)
    one = encode(dtype, np.ones(1))[0]
    sigma = SIGMA[dtype] if mode == "pin" else 0
    for lane, b, e in code_lane:
        Bp[:, lane] = encode(dtype, C[e].astype(np.float64) * 2.0**sigma * (1 if b % 2 == 0 else -1))
    for lane, s in noise_lane:
        Bp[:, lane] = encode(dtype, np.array([2.0**sigma]))[0]
    a1 = coins.integers(0, 2, (m, nb, CODE_LANES))
    at = coins.integers(0, 2, (m, S))
    # integer noise of each step's word: the block's code term plus the block's later noise lanes so far
    noise = np.zeros((m, n, S), np.int64)
    # block b's code lanes carry (-1)^b C, so the code term's mean, sum_e C[e, j] / 2 a block, cancels over block pairs
    alt = np.where(np.arange(nb) % 2 == 0, 1, -1)
    code_term = np.einsum("mbe,en->mnb", a1, C) * alt[None, None, :]
    first = np.r_[True, step_block[1:] != step_block[:-1]]
    run = np.zeros((m,), np.int64)
    for s in range(S):
        if first[s]:
            run = np.zeros((m,), np.int64)
        else:
            run = run + at[:, s]
        noise[:, :, s] = code_term[:, :, step_block[s]] + run[:, None]
    if mode == "fixed":
        u = np.ones(m)
        uw = np.full(m, one)
        beta = np.rint((C.sum(0) + S - 1) / 2.0)
        bias = np.broadcast_to(-beta[None, :, None], (m, n, nb)).astype(np.float64)
        epi = np.zeros_like(bias)
        p = None
    elif mode == "pin":
        fb = frac_bits(dtype)
        vals_a, vals_b = decode(dtype, A), decode(dtype, B)
        if bound == "cs":
            dmax = np.zeros(m)
            for b in range(nb):
                ks = dk[step_block[dl // p_.k] == b]
                na = np.linalg.norm(vals_a[:, ks], axis=1)
                nbn = np.linalg.norm(vals_b[:, ks], axis=1)
                dmax = np.maximum(dmax, na * nbn.max())
        elif bound == "partial":
            prod = vals_a[:, None, :] * vals_b[None, :, :]
            dmax = np.abs(prod).max(axis=(1, 2))
            for b in range(nb):
                ks = dk[step_block[dl // p_.k] == b]
                cs = np.cumsum(prod[:, :, ks], axis=-1)
                dmax = np.maximum(dmax, np.abs(cs).max(axis=(1, 2)))
        else:
            raise ValueError(bound)
        steps_in_block = np.bincount(step_block).max()
        nmax = CODE_LANES * 6 + steps_in_block
        # 2^p > 2 (dmax + nmax u) with u = 2^(p - fb): p = ceil(log2(2 dmax / (1 - 2 nmax 2^-fb)))
        need = 2 * dmax / (1 - 2 * nmax * 2.0**-fb)
        p = np.ceil(np.log2(np.maximum(need, 2.0**-60))).astype(np.int64)
        lo, hi = POW2[dtype]
        # a row too small for the format's least power of two takes a larger unit; one too large for its largest can't be
        # pinned in one block (`unpinnable`: its words leave the binade, the drawn block's Program refuses them)
        ra = np.clip(p - fb - sigma, lo, hi)
        unpinnable = (p - fb - sigma) > hi
        p = ra + sigma + fb
        u = np.exp2((ra + sigma).astype(np.float64))
        uw = pow2_word(dtype, ra)
        if not np.array_equal(decode(dtype, uw), np.exp2(ra.astype(np.float64))):
            raise ValueError("noise lane outside the format's exact powers of two")
        centre = np.rint((S / nb - 1) / 2.0)
        epi = np.broadcast_to((1.5 * np.exp2(p.astype(np.float64)))[:, None, None], (m, n, nb)).copy()
        bias = epi - (centre * u)[:, None, None]
    else:
        raise ValueError(mode)
    for lane, b, e in code_lane:
        Ap[:, lane] = np.where(a1[:, b, e] == 1, uw, 0)
    for lane, s in noise_lane:
        Ap[:, lane] = np.where(at[:, s] == 1, uw, 0)
    init = bias.astype(np.float32).view(np.uint32)
    if not np.array_equal(init.view(np.float32).astype(np.float64), bias):
        raise ValueError("accumulator init not exact in FP32")
    return Formed(Ap, Bp, init, u, noise, S, nb, step_block, bias, epi, p, unpinnable if mode == "pin" else None)


def run(dtype: str, f: Formed, chunk_rows: int = 16):
    """Every step's word (m, n, S) of the formed call, each block chained from its own init; the out-of-domain flag."""
    p_ = pipe_of(dtype)
    m, n = f.A.shape[0], f.B.shape[0]
    out = np.zeros((m, n, f.steps), np.uint32)
    bad = np.zeros((m, n), bool)
    for r0 in range(0, m, chunk_rows):
        rs = slice(r0, r0 + chunk_rows)
        a = f.A[rs][:, None, :]
        b = f.B[None, :, :]
        mm = a.shape[0]
        acc = None
        for s in range(f.steps):
            blk = f.step_block[s]
            if s == 0 or blk != f.step_block[s - 1]:
                acc = f.init[rs, :, blk]
            sl = slice(s * p_.k, (s + 1) * p_.k)
            acc, bb = sim.step(p_, acc, np.broadcast_to(a[..., sl], (mm, n, p_.k)), np.broadcast_to(b[..., sl], (mm, n, p_.k)))
            out[rs, :, s] = acc
            bad[rs] |= bb
    return out, bad


def _last(f: Formed) -> np.ndarray:
    return np.r_[f.step_block[1:] != f.step_block[:-1], True]


def served(dtype: str, f: Formed, words: np.ndarray) -> np.ndarray:
    """The served value of each output: the sum over blocks of (block's last word - the epilogue's constant), float64."""
    v = sim.fp32_value(words[:, :, _last(f)])
    return (v - f.epi).sum(-1)


def served_noise(f: Formed) -> np.ndarray:
    """The noise the served value carries (pin: exactly; fixed: as formed, before absorption)."""
    return (f.noise[:, :, _last(f)] * f.u[:, None, None] + f.bias - f.epi).sum(-1)
