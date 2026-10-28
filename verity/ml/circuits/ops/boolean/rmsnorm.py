"""A bf16 row's RMSNorm as Boolean Definitions: one proof block a row where the row fits a proof block, and the e2e demo's two-level
form where it doesn't. Every output fits its size class: 16 bits a bf16 word, 32 an f32.

    RmsNormRow_v1{N, EPS}       (x: Array<N, BF16>, w: Array<N, BF16>) -> Array<N, BF16>
    RmsSumSqPart_v1{C}          (x: Array<C, BF16>) -> F32
    F32SumPart_v1{G}            (s: Array<G, F32>) -> F32
    RmsNormRecip_v1{N, G, EPS}  (s: Array<G, F32>) -> F32
    RmsNormOut_v1{C}            (x: Array<C, BF16>, rs: F32, w: Array<C, BF16>) -> Array<C, BF16>
    RmsSumSqChunks_v1{C, K}     (x: Array<C, BF16>) -> Array<C/K, F32>
    RmsNormOutSums_v1{C, S, G, N, EPS}  (x: Array<C, BF16>, w: Array<C, BF16>, s: Array<S, F32>) -> Array<C, BF16>

(BF16 is ``Array<16, Value<1>>``, F32 ``Array<32, Value<1>>``.) `RmsNormRow`'s function is `RMSNormRef_v1{N, EPS, bf16}`'s
(`verity.ml.definitions.registry.ref_prims`, HF's ``LlamaRMSNorm``) on the GPU's operations, so a kernel matches it bit for bit
with ``mul.rn.f32``, ``add.rn.f32``, ``div.rn.f32``, ``sqrt.rn.f32`` and ``cvt.rn.bf16.f32``:

- ``acc = (((+0 + RN(x_0²)) + RN(x_1²)) + ...) + RN(x_{N-1}²)``, each x_i widened to f32 exactly, the sum in ascending order;
- ``rs = RN(1 / RN(√RN(RN(acc / N) + e)))``, e the f32 nearest EPS (`numpy.float32`);
- ``out_j = bf16_rn(RN(f32(w_j) * f32(bf16_rn(RN(f32(x_j) * rs)))))``: `Bf16MulBf16_v2` (w_j, `Bf16MulF32_v2` (x_j, rs)).

Every f32 operation's NaN is the GPU's 0x7FFFFFFF and every bf16 rounding F2FP's (NaN 0x7FFF), as in `elementwise`'s `_v4`s;
nothing is flushed. A NaN in the row makes rs a NaN and so every output 0x7FFF; `RMSNormRef_v1`'s outputs differ from these
only there (its NaN is torch's).

A long row doesn't fit one proof block: at N = 16,384 its three 32 KiB rows are 768 SHA-512 compressions, 2^26 bits of C-Flock's
block by themselves, and its 3,364 ANDs an element another 2^26, past the 2^27 a block holds. There the two-level form computes the same steps with the sum's grouping stated: `RmsSumSqPart` sums a chunk's squares from +0 in ascending
order, `F32SumPart` sums a group of those chunk sums from +0 in ascending order, `RmsNormRecip` sums the G group sums from +0
in ascending order and takes rs from that, and `RmsNormOut` writes a chunk of the output. At N = 16,384 the chunks are 128
words and the groups 64 chunk sums, so G = 2: 128 + 2 + 1 + 128 proof blocks a row. With one chunk and G = 1 it is `RmsNormRow`'s
function (+0 + s = s for every s it sums). An f32 sum is not associative, so another grouping is another Definition's
binding, never a regrouping of these.

C-Flock refuses a proof block none of whose rows is over 111 bytes, which `RmsNormRecip` (one 64-bit row) and `F32SumPart` (wired to
G four-byte rows) are. `RmsNormOutSums` folds them into the output: it is
``RmsNormOut{C}(x, RmsNormRecip{N, G, EPS}(F32SumPart{S/G} of each run of S/G consecutive s), w)`` word for word, so it
reads x, w and all S chunk sums. Its s is one parameter, so a statement may lay it out one sum a row (128 four-byte rows at N
= 16,384, 131 rows in all and so 256 hm96 slots) or several a row, as `RmsSumSqChunks` writes them: C/K chunk sums, chunk i
`RmsSumSqPart{K}` of words iK to iK + K - 1.

The square root is `f32_sqrt_rn`, ``PouwF32SqrtRn_v1``'s builder (PoUW's `PouwF32SqrtRn_v2` traces it in
`verity.ml.protocols.pouw.circuit.boolean`, which the catalog can't import): `isqrt` and `f32_sqrt_rn` here are copies of
it. `ops.boolean.sqrt` holds the same builder, which replaces these copies (``from .sqrt import f32_sqrt_rn``).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.types import Array
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, const_bits, lzc, mux, sub

from .. import fp32 as W
from .elementwise import bf16_mul_f32, specialized

__all__ = ["F32SumPart", "RmsNormOut", "RmsNormOutSums", "RmsNormRecip", "RmsNormRow", "RmsSumSqChunks", "RmsSumSqPart",
           "f32_bits", "f32_sqrt_rn", "f32_sum", "isqrt", "recip_root", "rms_norm_out", "rms_norm_out_sums", "rms_norm_row",
           "sum_squares"]

BF16 = Array(16, BL.BIT)
F32 = Array(32, BL.BIT)
ONE_F = 0x3F800000
_NAN = fp.F32_NAN_PTX
_F32_MAX = float(np.finfo(np.float32).max)


def isqrt(C, n):
    """⌊√n⌋ of an unsigned word of even width (LSB first), half as wide: one trial subtraction per root bit."""
    rem, root = [], []
    for i in range(len(n) // 2 - 1, -1, -1):
        rem = [n[2 * i], n[2 * i + 1]] + rem
        trial = [ONE, ZERO] + root
        w = max(len(rem), len(trial))
        d, borrow = sub(C, rem, trial, w)
        bit = NOT(borrow)
        rem = mux(C, bit, d, rem + [ZERO] * (w - len(rem)))[:len(root) + 2]
        root = [bit] + root
    return root


def f32_sqrt_rn(C, a):
    """``PouwF32SqrtRn_v1``: sqrt.rn.f32, √±0 = ±0, √+inf = +inf, a NaN or a negative nonzero operand 0x7FFFFFFF."""
    A = fp.unpack(C, a)
    _az, lz = lzc(C, [ZERO] * 8 + A.sig)
    sig = fp.shl(C, A.sig, lz)                                          # 1.f, the top bit set
    x = sub(C, add(C, A.e_eff, const_bits(127, 9), width=9), lz, 9)[0]  # e + 254 for the unbiased exponent e
    m = mux(C, x[0], [ZERO] + sig, sig + [ZERO])                        # e odd: 2 · 1.f and e - 1
    r = isqrt(C, [ZERO] * 25 + m)                                       # 25 bits, the top one set
    word = add(C, r[1:24] + x[1:9], [r[0]], width=31) + [ZERO]          # exponent field ⌊e / 2⌋ + 127
    keep = OR(C, A.zero, C.AND(A.inf, NOT(A.s)))
    nan = C.AND(NOT(keep), OR(C, A.e_max, A.s))
    return mux(C, nan, const_bits(_NAN, 32), mux(C, keep, list(a), word))


def f32_bits(v: float) -> int:
    """The f32 word nearest ``v`` (`numpy.float32`, ties to even), as the reference rounds EPS and N."""
    return W.bits_of(np.float32(v))


def f32_sum(C, words):
    """The f32 words summed from +0 in ascending order, ``RN(RN(+0 + s_0) + s_1) ...``."""
    acc = fp.f32_const(0)
    for s in words:
        acc = fp.f32_add(C, acc, s, nan_word=_NAN)
    return acc


def sum_squares(C, xs):
    """`f32_sum` of ``RN(f32(x_i)²)`` over the bf16 words ``xs``."""
    return f32_sum(C, [fp.f32_mul(C, f, f, nan_word=_NAN) for f in map(fp.bf16_to_f32, xs)])


def recip_root(C, acc, n: int, eps: int):
    """``rs = RN(1 / RN(√RN(RN(acc / n) + eps)))``, ``eps`` an f32 word."""
    mean = fp.f32_div(C, acc, fp.f32_const(f32_bits(n)), nan_word=_NAN)
    root = f32_sqrt_rn(C, fp.f32_add(C, mean, fp.f32_const(eps), nan_word=_NAN))
    return fp.f32_div(C, fp.f32_const(ONE_F), root, nan_word=_NAN)


def rms_norm_out(C, xs, rs, ws):
    """``out_j = bf16_rn(RN(f32(w_j) * f32(bf16_rn(RN(f32(x_j) * rs)))))`` for each x_j, w_j."""
    return [bf16_mul_f32(C, w, fp.bf16_to_f32(bf16_mul_f32(C, x, rs))) for x, w in zip(xs, ws, strict=True)]


def rms_norm_row(C, xs, ws, eps: int):
    """`rms_norm_out` at rs = `recip_root` of the row's `sum_squares`."""
    return rms_norm_out(C, xs, recip_root(C, sum_squares(C, xs), len(xs), eps), ws)


def rms_norm_out_sums(C, xs, ss, ws, groups: int, n: int, eps: int):
    """`rms_norm_out` at rs = `recip_root` of the `f32_sum` of ``groups`` group sums, group g the `f32_sum` of ``ss``'s g-th
    run of len(ss) / groups consecutive words."""
    k = len(ss) // groups
    acc = f32_sum(C, [f32_sum(C, ss[g * k:(g + 1) * k]) for g in range(groups)])
    return rms_norm_out(C, xs, recip_root(C, acc, n, eps), ws)


def _count(S: Any, k: str, what: str) -> int:
    v = getattr(S, k)
    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
        raise ValueError(f"{k} is {what}, at least 1, not {v!r}")
    return v


def _eps(S: Any) -> int:
    v = S.EPS
    if not isinstance(v, float) or not 0 < v <= _F32_MAX or f32_bits(v) == 0:
        raise ValueError(f"EPS is a positive float whose f32 is finite and nonzero, not {v!r}")
    return f32_bits(v)


_N = lambda S: _count(S, "N", "the row's length")  # noqa: E731
_C = lambda S: _count(S, "C", "the chunk's length")  # noqa: E731
_G = lambda S: _count(S, "G", "the number of sums")  # noqa: E731


def _row_build(S: Any):
    n, eps = _N(S), _eps(S)
    return lambda C, *e: rms_norm_row(C, e[:n], e[n:], eps)


def _row_signature(S: Any):
    _eps(S)
    row = Array(_N(S), BF16)
    return (("x", row), ("w", row)), row


def _sum_squares_build(S: Any):
    _C(S)
    return lambda C, *xs: sum_squares(C, xs)


def _sum_build(S: Any):
    _G(S)
    return lambda C, *s: f32_sum(C, s)


def _recip_build(S: Any):
    n, eps = _N(S), _eps(S)
    _G(S)
    return lambda C, *s: recip_root(C, f32_sum(C, s), n, eps)


def _recip_signature(S: Any):
    _N(S), _eps(S)
    return (("s", Array(_G(S), F32)),), F32


def _out_build(S: Any):
    c = _C(S)
    return lambda C, *e: rms_norm_out(C, e[:c], e[c], e[c + 1:])


def _out_signature(S: Any):
    chunk = Array(_C(S), BF16)
    return (("x", chunk), ("rs", F32), ("w", chunk)), chunk


def _per(S: Any, whole: str, part: str) -> int:
    a, b = _count(S, whole, f"a multiple of {part}"), _count(S, part, f"a divisor of {whole}")
    if a % b:
        raise ValueError(f"{whole} = {a} is not a multiple of {part} = {b}")
    return a // b


def _chunks_build(S: Any):
    _per(S, "C", "K")
    k = S.K
    return lambda C, *xs: [sum_squares(C, xs[i:i + k]) for i in range(0, len(xs), k)]


def _chunks_signature(S: Any):
    return (("x", Array(_C(S), BF16)),), Array(_per(S, "C", "K"), F32)


def _out_sums_build(S: Any):
    _per(S, "S", "G")
    c, g, n, eps = _C(S), S.G, _N(S), _eps(S)
    return lambda C, *e: rms_norm_out_sums(C, e[:c], e[2 * c:], e[c:2 * c], g, n, eps)


def _out_sums_signature(S: Any):
    _per(S, "S", "G"), _N(S), _eps(S)
    chunk = Array(_C(S), BF16)
    return (("x", chunk), ("w", chunk), ("s", Array(S.S, F32))), chunk


RmsNormRow = specialized("RmsNormRow", 1, ("N", "EPS"), _row_build, _row_signature,
                         doc="a bf16 row's RMSNorm on bits, one unit a row: bf16_rn(RN(w_j * bf16_rn(RN(x_j * rs)))), "
                             "rs = RN(1 / RN(sqrt(RN(RN(acc / N) + EPS)))), acc the RN(x_i^2) summed from +0 in ascending order; "
                             "every f32 NaN 0x7FFFFFFF, every bf16 NaN 0x7FFF")
RmsSumSqPart = specialized("RmsSumSqPart", 1, ("C",), _sum_squares_build, lambda S: ((("x", Array(_C(S), BF16)),), F32),
                           doc="a chunk of a bf16 row's squares on bits: RN(x_i^2) summed from +0 in ascending order, "
                               "every NaN 0x7FFFFFFF")
F32SumPart = specialized("F32SumPart", 1, ("G",), _sum_build, lambda S: ((("s", Array(_G(S), F32)),), F32),
                         doc="G f32 words summed from +0 in ascending order on bits, every NaN 0x7FFFFFFF")
RmsNormRecip = specialized("RmsNormRecip", 1, ("N", "G", "EPS"), _recip_build, _recip_signature,
                           doc="RMSNorm's reciprocal root on bits from a row's G partial sums of squares: "
                               "RN(1 / RN(sqrt(RN(RN(acc / N) + EPS)))), acc the G sums summed from +0 in ascending order; "
                               "every NaN 0x7FFFFFFF")
RmsNormOut = specialized("RmsNormOut", 1, ("C",), _out_build, _out_signature,
                         doc="a chunk of RMSNorm's output on bits: bf16_rn(RN(w_j * bf16_rn(RN(x_j * rs)))), NaN 0x7FFF")
RmsSumSqChunks = specialized("RmsSumSqChunks", 1, ("C", "K"), _chunks_build, _chunks_signature,
                             doc="C/K chunks of a bf16 row's squares on bits, chunk i RN(x_j^2) for j in [iK, iK + K) summed "
                                 "from +0 in ascending order, every NaN 0x7FFFFFFF")
RmsNormOutSums = specialized("RmsNormOutSums", 1, ("C", "S", "G", "N", "EPS"), _out_sums_build, _out_sums_signature,
                             doc="a chunk of RMSNorm's output on bits from the row's S chunk sums: "
                                 "bf16_rn(RN(w_j * bf16_rn(RN(x_j * rs)))), rs = RN(1 / RN(sqrt(RN(RN(acc / N) + EPS)))), "
                                 "acc the G group sums summed from +0, group g s's g-th run of S/G words summed from +0, "
                                 "both in ascending order; every f32 NaN 0x7FFFFFFF, every bf16 NaN 0x7FFF")
