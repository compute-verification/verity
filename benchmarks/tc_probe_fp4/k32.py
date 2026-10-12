"""The K=32 E2M1 rows on sm_120 (``kind::f8f6f4`` and ``kind::mxf8f6f4 .scale_vec::1X`` with an E2M1 operand): encodings, the
candidate models the capture decides between, and the families that separate them.

Both kinds lower to the QMMA datapath (SASS ``QMMA.16832.F32.<a>.<b>``, ``QMMA.SF.16832.F32.<a>.<b>.E8``), the one the vLLM port
measured for E4M3 x E4M3 on the RTX PRO 6000 (``GroupSum`` groups (32,), width 26, floor -133; lane ``vllm-sm120-tc-gemm``), while
``kind::mxf4nvf4`` lowers to OMMA, whose step is ``BlockScaledAlignAdd``.  Which of the two an E2M1 operand gets on QMMA is what the
capture decides; nothing here is a validated model, every candidate is a hypothesis with its name.

Candidates (``candidates(instr)``):

* ``gs32_w26_native``: ``GroupSum`` (32,)/26/-133 on the exact mixed products, each E2M1 product carried like
  ``term.Format.product`` does (a subnormal E2M1 operand, 0.5, keeps its subnormal alignment exponent);
* ``gs32_w26_as_e4m3``: the same with every E2M1 operand first converted to its E4M3 byte (exact; 0.5 becomes a normal number, so
  its product term sits one binade lower on the alignment grid): the "one FP8 datapath behind a converter" hypothesis;
* ``gs16x2_w26_native``: two groups of 16 (Ada's structure, at width 26);
* ``bsaa_g16`` / ``bsaa_g32`` (E2M1 x E2M1 only): ``BlockScaledAlignAdd`` at k=32 with the nvf4 adder constants (frac_bits 27,
  acc_window 36, grid_floor -174), unscaled or with the one UE8M0 byte of ``.scale_vec::1X``: the "OMMA adder on QMMA" hypothesis.

A UE8M0 scale pair (``mxf8f6f4``) enters the GroupSum candidates as an exponent shift ``(sa - 127) + (sb - 127)`` of every product
before alignment, and ``BlockScaledAlignAdd`` as its own lossless scale.  The captured tiles are published whole
(``tiles_<family>.npz``), so any further candidate is fitted offline without the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from verity.ml.circuits.silicon.errors import InvalidArtifact
from verity.ml.circuits.silicon import models as M
from verity.ml.circuits.silicon.term import FORMATS, FP32_MAX_EXPONENT, PRODUCT_WIDTH, ZERO, Term, fp32_term, pack_fp32

E2M1_VALUE = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)
#: E2M1 code -> the E4M3 byte of the same value (every E2M1 value is an E4M3 normal or zero)
E2M1_TO_E4M3 = [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C, 0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC]
E4M3_NAN = (0x7F, 0xFF)


def _e4m3_values() -> np.ndarray:
    out = np.zeros(256, np.float64)
    for w in range(256):
        e, m, s = (w >> 3) & 0xF, w & 7, -1.0 if w & 0x80 else 1.0
        out[w] = np.nan if (w & 0x7F) == 0x7F else s * (m * 2.0**-9 if e == 0 else (8 + m) * 2.0 ** (e - 10))
    return out


E4M3_VALUE = _e4m3_values()
VALUE = {"e2m1": E2M1_VALUE, "e4m3": E4M3_VALUE}
SIGN_BIT = {"e2m1": 0x8, "e4m3": 0x80}

#: instruction -> (format of A, format of B, scale) for the K=32 rows
ROWS = {
    "f8f6f4": ("e2m1", "e2m1", None),
    "f8f6f4_e2m1_e4m3": ("e2m1", "e4m3", None),
    "f8f6f4_e4m3_e2m1": ("e4m3", "e2m1", None),
    "mxf8f6f4_e2m1": ("e2m1", "e2m1", "ue8m0"),
    "mxf8f6f4_e2m1_e4m3": ("e2m1", "e4m3", "ue8m0"),
    "mxf8f6f4_e4m3_e2m1": ("e4m3", "e2m1", "ue8m0"),
}


def container(fmt: str, codes: np.ndarray) -> np.ndarray:
    """Element codes -> the 8-bit containers the f8f6f4 / mxf8f6f4 registers carry (E2M1 at bits [2,6); E4M3 as is)."""
    c = np.asarray(codes, dtype=np.uint8)
    return ((c & 0xF) << 2).astype(np.uint8) if fmt == "e2m1" else c


def mixed_product(fa: str, fb: str, a: int, b: int) -> Term:
    """The exact product of an ``fa`` word and an ``fb`` word as a 25-bit-scaled Term (``term.Format.product`` for mixed formats)."""
    FA, FB = FORMATS[fa], FORMATS[fb]
    da, db = FA.decode(a), FB.decode(b)
    if isinstance(da, str) or isinstance(db, str):
        raise InvalidArtifact("non-finite operand")
    (sga, ea, ma), (sgb, eb, mb) = da, db
    if ma == 0 or mb == 0:
        return ZERO
    shift = PRODUCT_WIDTH - (FA.sig_bits + FB.sig_bits)
    return Term(bool(sga ^ sgb), ea + eb - shift + 23, (ma * mb) << shift)


@dataclass(frozen=True)
class GroupSumCandidate:
    """``GroupSum`` on mixed-format products, optionally with every E2M1 operand converted to E4M3 first and a UE8M0 pair."""

    name: str
    fmt_a: str
    fmt_b: str
    groups: tuple[int, ...]
    width: int
    zero_exponent: int
    e2m1_as_e4m3: bool = False
    scale: str | None = None
    _table: dict = field(default_factory=dict, compare=False, repr=False)

    def _product(self, a: int, b: int) -> Term:
        key = (a, b)
        t = self._table.get(key)
        if t is None:
            fa, fb = self.fmt_a, self.fmt_b
            if self.e2m1_as_e4m3:
                if fa == "e2m1":
                    a, fa = E2M1_TO_E4M3[a & 0xF], "e4m3"
                if fb == "e2m1":
                    b, fb = E2M1_TO_E4M3[b & 0xF], "e4m3"
            t = self._table[key] = mixed_product(fa, fb, a, b)
        return t

    def step_scaled(self, c_word: int, a_codes, b_codes, a_scales=(), b_scales=()) -> int:
        if (c_word >> 23) & 0xFF == 0xFF:
            raise InvalidArtifact("non-finite FP32 accumulator")
        shift = 0
        if self.scale:
            if a_scales[0] == 0xFF or b_scales[0] == 0xFF:
                raise InvalidArtifact("UE8M0 NaN scale")
            shift = int(a_scales[0]) + int(b_scales[0]) - 254
        acc = fp32_term(c_word)
        start = 0
        for size in self.groups:
            terms = [acc]
            for i in range(start, start + size):
                t = self._product(int(a_codes[i]), int(b_codes[i]))
                if t.magnitude and shift:
                    t = Term(t.negative, t.exponent + shift, t.magnitude)
                terms.append(t)
            acc = M.group_sum(terms, width=self.width, zero_exponent=self.zero_exponent)
            if acc.magnitude and acc.exponent > FP32_MAX_EXPONENT:
                raise InvalidArtifact("sum leaves the binary32 range (not modelled)")
            start += size
        return pack_fp32(acc)


def _bsaa(name: str, scale: str | None, group: int) -> M.BlockScaledAlignAdd:
    nv = M.BLACKWELL_SM120_NVF4
    return M.BlockScaledAlignAdd(name=name, arch="sm_120", dtype="e2m1", operand_bits=4, scale=scale, k_per_instruction=32, group=group,
                                 frac_bits=nv.frac_bits, acc_window=nv.acc_window, grid_floor=nv.grid_floor, underflow_zero="positive",
                                 validated="probe candidate: the nvf4 adder at k=32 (hypothesis)")


def candidates(instr: str) -> dict:
    """The candidate models of a K=32 row, by name (see the module docstring)."""
    fa, fb, scale = ROWS[instr]
    out = {
        "gs32_w26_native": GroupSumCandidate("gs32_w26_native", fa, fb, (32,), 26, -133, False, scale),
        "gs32_w26_as_e4m3": GroupSumCandidate("gs32_w26_as_e4m3", fa, fb, (32,), 26, -133, True, scale),
        "gs16x2_w26_native": GroupSumCandidate("gs16x2_w26_native", fa, fb, (16, 16), 26, -133, False, scale),
    }
    if fa == fb == "e2m1":
        out["bsaa_g16"] = _bsaa(f"probe_{instr}_bsaa_g16", scale, 16)
        out["bsaa_g32"] = _bsaa(f"probe_{instr}_bsaa_g32", scale, 32)
    return out


# --- families --------------------------------------------------------------------------------------------------------------------


def _codes(rng, fmt: str, shape, nonzero: bool = False) -> np.ndarray:
    if fmt == "e2m1":
        c = rng.integers(1 if nonzero else 0, 16, shape)
        if nonzero:
            c = np.where(c == 8, 1, c)
        return c.astype(np.uint8)
    c = rng.integers(1 if nonzero else 0, 256, shape)
    c = np.where(np.isin(c, E4M3_NAN), c - 1, c)
    if nonzero:
        c = np.where(c == 0x80, 0x81, c)
    return c.astype(np.uint8)


def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def _acc_near(rng, p: np.ndarray, d_lo: int, d_hi: int) -> np.ndarray:
    """FP32 words +-(1.f) 2^(floor(log2|p|) + d), d uniform in [d_lo, d_hi], the mantissa exact / all-ones / random in thirds."""
    n = p.size
    lp = np.floor(np.log2(np.where(p == 0, 1.0, np.abs(p)))).astype(np.int64).ravel()
    e = np.clip(lp + rng.integers(d_lo, d_hi + 1, n) + 127, 1, 254).astype(np.uint32)
    kind = rng.integers(0, 3, n)
    f = np.where(kind == 0, 0, np.where(kind == 1, (1 << 23) - 1, rng.integers(0, 1 << 23, n))).astype(np.uint32)
    s = rng.integers(0, 2, n).astype(np.uint32)
    return ((s << 31) | (e << 23) | f).reshape(p.shape)


def families_k32(rng, n_random: int, instr: str, fixed_mult: int = 1):
    """Yield (name, A (n,16,32) codes, Bt (n,8,32), SFA (n,16,1) | None, SFB (n,8,1) | None, C (n,16,8)) for a K=32 row.
    ``fixed_mult`` multiplies the one gated family of fixed size (``signed_zero``, 256 tiles); 1 keeps every earlier stream."""
    fa, fb, scale = ROWS[instr]
    k = 32
    va, vb = VALUE[fa], VALUE[fb]

    def scales(shape, lo=117, hi=138):
        return rng.integers(lo, hi, shape).astype(np.uint8) if scale else None

    def sval(s, n, rows):
        return np.ones((n, rows)) if s is None else np.exp2(s[..., 0].astype(np.float64) - 127)

    def acc_random(n, lo=-30, hi=40):
        return _f32((rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(lo, hi, (n, 16, 8)))).astype(np.float32))

    def zeros(n):
        return np.zeros((n, 16, 8), np.uint32)

    n = n_random
    yield "random_random_acc", _codes(rng, fa, (n, 16, k)), _codes(rng, fb, (n, 8, k)), scales((n, 16, 1)), scales((n, 8, 1)), acc_random(n)
    n = n_random // 2
    yield "random_zero_acc", _codes(rng, fa, (n, 16, k)), _codes(rng, fb, (n, 8, k)), scales((n, 16, 1)), scales((n, 8, 1)), zeros(n)
    n = n_random // 4
    yield ("uniform_encodings", _codes(rng, fa, (n, 16, k)), _codes(rng, fb, (n, 8, k)), scales((n, 16, 1), 0, 178),
           scales((n, 8, 1), 0, 178), _uniform_acc(rng, n))

    # the pipeline discriminators: one product per element (every row of A and column of B live at one k), the accumulator walked
    # across every candidate's alignment edge (-45 .. +30 binades from the product), both signs; the first tile holds the two
    # vectors that refute every GroupSum against the block-scaled rule on E2M1 x E2M1 (acc -2^-30 with 0.5 x 0.5, acc -2^-28 with
    # 6 x 6), and a sub-family restricts E2M1 operands to +-0.5 (the subnormal code, where the two E2M1 carry conventions differ)
    for name, sub in (("pipeline_discriminators", False), ("subnormal_e2m1_walk", True)):
        n = n_random // 2 if not sub else n_random // 8
        A = np.zeros((n, 16, k), np.uint8)
        Bt = np.zeros((n, 8, k), np.uint8)
        k0 = rng.integers(0, k, n)
        ca = _codes(rng, fa, (n, 16), nonzero=True)
        cb = _codes(rng, fb, (n, 8), nonzero=True)
        if sub:
            if fa == "e2m1":
                ca = np.where(rng.random((n, 16)) < 0.5, 1, 9).astype(np.uint8)
            if fb == "e2m1":
                cb = np.where(rng.random((n, 8)) < 0.5, 1, 9).astype(np.uint8)
        A[np.arange(n), :, k0] = ca
        Bt[np.arange(n), :, k0] = cb
        sfa, sfb = scales((n, 16, 1)), scales((n, 8, 1))
        p = va[ca][:, :, None] * vb[cb][:, None, :] * sval(sfa, n, 16)[:, :, None] * sval(sfb, n, 8)[:, None, :]
        C = _acc_near(rng, p, -45, 30)
        if not sub and fa == fb == "e2m1" and n >= 2:
            A[:2] = 0
            Bt[:2] = 0
            A[0, :, 0], Bt[0, :, 0], C[0] = 1, 1, 0xB0800000      # 0.5 x 0.5 = 0.25 onto -2^-30
            A[1, :, 0], Bt[1, :, 0], C[1] = 7, 7, 0xB1800000      # 6 x 6 = 36 onto -2^-28
            if scale:
                sfa[:2], sfb[:2] = 127, 127
        yield name, A, Bt, sfa, sfb, C

    # two products per element at different magnitudes, the accumulator near the larger one
    n = n_random // 4
    A = np.zeros((n, 16, k), np.uint8)
    Bt = np.zeros((n, 8, k), np.uint8)
    k0 = rng.integers(0, k, n)
    k1 = (k0 + rng.integers(1, k, n)) % k
    for kk in (k0, k1):
        A[np.arange(n), :, kk] = _codes(rng, fa, (n, 16), nonzero=True)
        Bt[np.arange(n), :, kk] = _codes(rng, fb, (n, 8), nonzero=True)
    sfa, sfb = scales((n, 16, 1)), scales((n, 8, 1))
    pa = np.abs(va[A.astype(np.int64)])
    pb = np.abs(vb[Bt.astype(np.int64)])
    pmax = np.einsum("nik,njk->nijk", pa, pb).max(-1) * sval(sfa, n, 16)[:, :, None] * sval(sfb, n, 8)[:, None, :]
    yield "two_product_walk", A, Bt, sfa, sfb, _acc_near(rng, pmax, -40, 10)

    # exact cancellation of k 0..15 against k 16..31 with a few residual terms, small accumulators
    n = n_random // 4
    A = _codes(rng, fa, (n, 16, k))
    Bt = _codes(rng, fb, (n, 8, k))
    A[:, :, 16:] = A[:, :, :16]
    Bt[:, :, 16:] = Bt[:, :, :16] ^ SIGN_BIT[fb]
    res = rng.random((n, 8, 16)) < 0.1
    Bt[:, :, 16:] = np.where(res, _codes(rng, fb, (n, 8, 16)), Bt[:, :, 16:])
    acc = zeros(n)
    acc[n // 2:] = acc_random(n - n // 2, -40, -10)
    yield "cancellation", A, Bt, scales((n, 16, 1)), scales((n, 8, 1)), acc

    n = n_random // 8
    yield "acc_window", _codes(rng, fa, (n, 16, k)), _codes(rng, fb, (n, 8, k)), scales((n, 16, 1)), scales((n, 8, 1)), acc_random(n, 20, 60)
    n = n_random // 8
    yield ("tiny_acc", _codes(rng, fa, (n, 16, k)), _codes(rng, fb, (n, 8, k)), scales((n, 16, 1)), scales((n, 8, 1)),
           _uniform_acc(rng, n, 0, 30))

    n = 256 * fixed_mult
    A = np.where(rng.random((n, 16, k)) < 0.5, 0, SIGN_BIT[fa]).astype(np.uint8)
    Bt = _codes(rng, fb, (n, 8, k))
    one_a = 2 if fa == "e2m1" else 0x38
    three_b = (5, 13) if fb == "e2m1" else (0x44, 0xC4)
    A[n // 2:, :, :16] = one_a
    Bt[n // 2:, :, :16] = np.where(np.arange(16) % 2 == 0, three_b[0], three_b[1]).astype(np.uint8)
    acc = np.where(rng.random((n, 16, 8)) < 0.5, np.uint32(0x80000000), np.uint32(0)).astype(np.uint32)
    yield "signed_zero", A, Bt, scales((n, 16, 1)), scales((n, 8, 1)), acc

    if scale:
        # every UE8M0 code on A (B at 1.0), then both swept with the pair's exponent sum capped so no sum leaves binary32
        n = n_random // 4
        sa = ((np.arange(n)[:, None] * 16 + np.arange(16)[None, :]) % 255).astype(np.int64)
        sb = np.full((n, 8), 127, np.int64)
        h = n // 2
        sb[h:] = (np.arange(n - h)[:, None] * 8 + np.arange(8)[None, :] + 97) % 255
        cap = 360 - sa.max(axis=1, keepdims=True)
        sb = np.minimum(sb, cap)
        acc = zeros(n)
        acc[n // 2:] = acc_random(n - n // 2)
        yield ("scale_sweep", _codes(rng, fa, (n, 16, k)), _codes(rng, fb, (n, 8, k)), sa.astype(np.uint8)[:, :, None],
               sb.astype(np.uint8)[:, :, None], acc)
        # scale extremes: sums far below the subnormal grid, the grid floor, subnormal accumulators
        n = n_random // 8
        lo = np.concatenate([np.arange(0, 12), np.arange(100, 112)])
        sfa = lo[rng.integers(0, len(lo), (n, 16, 1))].astype(np.uint8)
        sfb = lo[rng.integers(0, len(lo), (n, 8, 1))].astype(np.uint8)
        acc = zeros(n)
        acc[n // 3: 2 * n // 3] = _uniform_acc(rng, 2 * n // 3 - n // 3, 0, 10)
        acc[2 * n // 3:] = acc_random(n - 2 * n // 3, -140, -100)
        yield "scale_extremes", _codes(rng, fa, (n, 16, k)), _codes(rng, fb, (n, 8, k)), sfa, sfb, acc

    # record-only: NaN / inf accumulators, NaN E4M3 operands, the UE8M0 NaN scale, sums leaving binary32
    n = 64
    A = _codes(rng, fa, (n, 16, k))
    Bt = _codes(rng, fb, (n, 8, k))
    acc = acc_random(n)
    specials = np.array([0x7FC00000, 0xFFC00000, 0x7F800001, 0x7F800000, 0xFF800000, 0x7FFFFFFF], np.uint32)
    acc[: n // 2] = specials[rng.integers(0, len(specials), (n // 2, 16, 8))]
    if fa == "e4m3":
        A[n // 2: 3 * n // 4, :, 0] = 0x7F
    if fb == "e4m3":
        Bt[n // 2: 3 * n // 4, :, 1] = 0xFF
    sfa, sfb = scales((n, 16, 1)), scales((n, 8, 1))
    if scale:
        sfa[3 * n // 4:, :4] = 0xFF
        sfa[3 * n // 4:, 4:], sfb[3 * n // 4:] = 250, 250
    yield "specials", A, Bt, sfa, sfb, acc


def _uniform_acc(rng, n: int, e_lo: int = 1, e_hi: int = 250) -> np.ndarray:
    e = rng.integers(e_lo, e_hi + 1, (n, 16, 8)).astype(np.uint32)
    m = rng.integers(0, 1 << 23, (n, 16, 8)).astype(np.uint32)
    s = rng.integers(0, 2, (n, 16, 8)).astype(np.uint32)
    return (s << 31) | (e << 23) | m


# --- layout check operands -------------------------------------------------------------------------------------------------------

#: exact small values and their codes: E2M1 integers, E4M3 integers -8..8
SMALL = {
    "e2m1": {0: 0, 1: 2, 2: 4, 3: 5, 4: 6, 6: 7, -1: 10, -2: 12, -3: 13, -4: 14, -6: 15},
    "e4m3": {v: c for v, c in [(0, 0x00)] + [(s * m, (0x80 if s < 0 else 0) | c) for m, c in
                                            [(1, 0x38), (2, 0x40), (3, 0x44), (4, 0x48), (5, 0x4A), (6, 0x4C), (7, 0x4E), (8, 0x50)]
                                            for s in (1, -1)]},
}


def small_operands(rng, fmt: str, shape) -> tuple[np.ndarray, np.ndarray]:
    """(values, codes) of exact small integers of ``fmt``."""
    vals = np.array(sorted(SMALL[fmt]), dtype=np.int64)
    v = vals[rng.integers(0, len(vals), shape)]
    lut = SMALL[fmt]
    codes = np.vectorize(lambda x: lut[int(x)], otypes=[np.uint8])(v)
    return v.astype(np.float64), codes
