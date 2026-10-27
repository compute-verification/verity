"""The e2e demo's row ops as Boolean Definitions split so that each outputs at most 16 bits (a proof block's outputs, the
16-bit fallback forms): a bf16 row's softmax, the embedding's coordinate, and the 16-bit halves of RMSNorm's f32 parts.

    SoftmaxMax_v1{N}                   (x: Array<N, Array<16, Value<1>>>) -> Array<16, Value<1>>
    SoftmaxRecip_v1{N, HALF}           (x: Array<N, Array<16, Value<1>>>, m: Array<16, Value<1>>) -> Array<16, Value<1>>
    SoftmaxOut_v1                      (x, m: Array<16, Value<1>>, r: Array<32, Value<1>>) -> Array<16, Value<1>>
    EmbeddingCoordinate_v1{V}          (tok: Array<16, Value<1>>, col: Array<V, Array<16, Value<1>>>, pos: Array<16, Value<1>>)
                                       -> Array<16, Value<1>>
    RmsSumSqPartHalf_v1{C, HALF}       (x: Array<C, Array<16, Value<1>>>) -> Array<16, Value<1>>
    F32SumPartHalf_v1{G, HALF}         (s: Array<G, Array<32, Value<1>>>) -> Array<16, Value<1>>
    RmsNormRecipHalf_v1{N, G, EPS, HALF}  (s: Array<G, Array<32, Value<1>>>) -> Array<16, Value<1>>

Softmax over x_0 .. x_{N-1} is `SoftmaxRef_v1{N, bf16}`'s dataflow (`verity.ml.definitions.registry.ref_prims`) on the GPU's
operations, so a kernel matches it bit for bit with ``add.rn.f32``, ``mul.rn.f32``, ``div.rn.f32``, ``ex2.approx.ftz.f32``
and ``cvt.rn.bf16.f32``:

- ``m`` is the first greatest x_i (+0 and -0 one value, so the earlier is kept, as `F32MaxRef_v1`'s fold keeps it), or 0x7FFF
  when any x_i is a NaN. It is one of the x_i, so `SoftmaxMax` outputs it exactly, in 16 bits.
- ``e_j = MUFU.EX2.FTZ(RN(RN(f32(x_j) - f32(m)) * L))``, L = RN32(log2 e) = 0x3FB8AA3B (`MufuEx2Ftz_v2`'s builder): the
  reference's `F32ExpRn_v1` has no Boolean form, and MUFU.EX2 is what an sm_90 kernel computes, read from the sm_89 table
  as the H200 Definitions read it (`verity.ml.definitions.numerics.fa3_model`: assumed identical on sm_90).
- ``s = (((+0 + e_0) + e_1) + ...) + e_{N-1}``, in ascending order, and ``r = RN(1 / s)``. `SoftmaxRecip` outputs r's low
  (HALF = 0) or high (HALF = 1) 16 bits.
- ``p_j = bf16_rn(RN(e_j * r))``. `SoftmaxOut` recomputes e_j from x_j and m, so it reads x_j, m and r's two halves.

**The embedding's coordinate** j at step t is ``Bf16AddF2fp_v1(h, pos)``, where h is ``col[tok]`` if tok < V and 0x7FFF
otherwise, as `Embedding_v1` fills (`verity.ml.commitments.gates.embedding`). Here col is column j of the V × H token table,
pos is the position table's word (t, j), and tok is the u16 token that the previous step's `ArgmaxCombine_v1` or the request
row gives. col and pos are registered reads at public positions (j and t), and tok is a value, so the column is selected
inside the proof. The select is `MoeSelectBf16x{E}_v2`'s: the multiplexer tree (`boolean.gather`) over tok's low
b = ceil(log2 V) bits, with a word past V standing for 0x7FFF, then 0x7FFF wherever any of tok's other 16 - b bits is set.

**RMSNorm** is `ops.boolean.rmsnorm`'s two-level form. Its f32 parts each become two proof blocks, one for each 16-bit half: the low
16 bits (HALF = 0) or the high 16 bits (HALF = 1) of `RmsSumSqPart_v1{C}`, `F32SumPart_v1{G}` or `RmsNormRecip_v1{N, G, EPS}`.
Each half is traced from that part's builder, so a half's circuit holds only the gates its own bits read. Each takes the part's
statics and refuses what the part's signature refuses. A later part, or `RmsNormOut_v1{C}` (C = 1 is its 16-bit form), reads
an f32 as the two halves' rows. At Phase 1's N = 128 the row is one chunk with G = 1: 2 halves of the chunk's sum, then 2 of
rs. At Phase 2's N = 16,384, with chunks of 128 and groups of 64: 256 chunk halves, 4 group halves, then 2 of rs.

Every f32 operation's NaN is the GPU's 0x7FFFFFFF and every bf16 rounding F2FP's (NaN 0x7FFF), as in `elementwise`'s `_v4`s;
none is flushed but MUFU.EX2's operand. An f32 sum is not associative, so a split of a longer row is a new Definition that
states its order, never a regrouping of this one.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.types import Array
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.forms import OR, const_bits, mux, or_reduce
from verity.ml.circuits.boolean.gather import gather
from verity.ml.circuits.boolean.trace import definition

from .. import fp32 as W
from . import rmsnorm as R
from .elementwise import bf16_add_f2fp, specialized
from .mufu import mufu_ex2
from .scalar import bf16_key, gt_u

__all__ = ["EMBEDDING_FILL", "EmbeddingCoordinate", "F32SumPartHalf", "LOG2E", "RmsNormRecipHalf", "RmsSumSqPartHalf", "SoftmaxMax",
           "SoftmaxOut", "SoftmaxRecip", "embedding_coordinate", "softmax_exp", "softmax_max", "softmax_out", "softmax_recip"]

BF16 = U16 = Array(16, BL.BIT)
F32 = Array(32, BL.BIT)
#: RN32(log2 e)
LOG2E = 0x3FB8AA3B
ONE_F = 0x3F800000
_NAN = fp.F32_NAN_PTX
#: the word a token outside the vocabulary selects (`Embedding_v1`'s fill)
EMBEDDING_FILL = 0x7FFF
#: the widest vocabulary a 16-bit token indexes
MAX_V = 1 << 16


def softmax_max(C, xs):
    """``m``: the first greatest of the bf16 words ``xs`` in `bf16_key`'s order (+0 and -0 one key), or 0x7FFF when any is a
    NaN. The running word and its key are both selected, so each word's key is made once."""
    nan, key = bf16_key(C, xs[0])
    m = list(xs[0])
    for x in xs[1:]:
        n, k = bf16_key(C, x)
        take = gt_u(C, k, key)
        m, key, nan = mux(C, take, x, m), mux(C, take, k, key), OR(C, nan, n)
    return mux(C, nan, const_bits(W.BF16_NAN_F2FP, 16), m)


def softmax_exp(C, x, m):
    """``e = MUFU.EX2.FTZ(RN(RN(f32(x) - f32(m)) * L))`` of bf16 words x and m."""
    d = fp.f32_add(C, fp.bf16_to_f32(x), fp.f32_neg(fp.bf16_to_f32(m)), nan_word=_NAN)
    return mufu_ex2(C, fp.f32_mul(C, d, fp.f32_const(LOG2E), nan_word=_NAN))


def softmax_recip(C, xs, m):
    """``r = RN(1 / s)``, ``s`` the row's `softmax_exp` words summed from +0 in ascending order."""
    s = fp.f32_const(0)
    for x in xs:
        s = fp.f32_add(C, s, softmax_exp(C, x, m), nan_word=_NAN)
    return fp.f32_div(C, fp.f32_const(ONE_F), s, nan_word=_NAN)


def softmax_out(C, x, m, r):
    """``p = bf16_rn(RN(e * r))``, e = `softmax_exp` (x, m)."""
    return fp.f32_to_bf16(C, fp.f32_mul(C, softmax_exp(C, x, m), r, nan_word=_NAN), W.BF16_NAN_F2FP)


def embedding_coordinate(C, tok, col, pos):
    """``Bf16AddF2fp_v1(col[tok], pos)``, col[tok] read as 0x7FFF for tok >= len(col)."""
    b = (len(col) - 1).bit_length()
    h = gather(C, tok[:b], list(col), EMBEDDING_FILL)
    if tok[b:]:
        h = mux(C, or_reduce(C, tok[b:]), const_bits(EMBEDDING_FILL, 16), h)
    return bf16_add_f2fp(C, h, pos)


def _row(S: Any) -> int:
    if not isinstance(S.N, int) or S.N < 1:
        raise ValueError(f"N is the row's length, at least 1, not {S.N!r}")
    return S.N


def _half(S: Any) -> int:
    if S.HALF not in (0, 1):
        raise ValueError(f"HALF is 0 (r's low 16 bits) or 1 (its high 16 bits), not {S.HALF!r}")
    return S.HALF


def _recip(S: Any):
    lo = 16 * _half(S)
    return lambda C, *e: softmax_recip(C, e[:-1], e[-1])[lo:lo + 16]


def _recip_signature(S: Any):
    _half(S)
    return (("x", Array(_row(S), BF16)), ("m", BF16)), U16


def _vocabulary(S: Any) -> int:
    v = S.V
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= MAX_V:
        raise ValueError(f"V is the vocabulary, an int in [1, 2^16] (a 16-bit token indexes it), not {v!r}")
    return v


def _embedding(S: Any):
    _vocabulary(S)
    return lambda C, tok, *e: embedding_coordinate(C, tok, e[:-1], e[-1])


def _embedding_signature(S: Any):
    return (("tok", U16), ("col", Array(_vocabulary(S), BF16)), ("pos", BF16)), BF16


def _f32_half(part, f):
    """The builder and signature of the 16-bit half (HALF) of the f32 part ``part``, whose builder at statics S is ``f(S)``:
    ``part``'s own signature function refuses its statics, and its parameters are the half's."""
    def build(S: Any):
        part.signature(S)
        lo, g = 16 * _half(S), f(S)
        return lambda C, *e: g(C, *e)[lo:lo + 16]

    def signature(S: Any):
        _half(S)
        return part.signature(S)[0], U16
    return build, signature


SoftmaxMax = specialized("SoftmaxMax", 1, ("N",), lambda S: lambda C, *xs: softmax_max(C, xs),
                         lambda S: ((("x", Array(_row(S), BF16)),), BF16),
                         doc="a bf16 row's softmax max on bits: the first greatest x_i (+0 == -0), 0x7FFF if any is a NaN")
SoftmaxRecip = specialized("SoftmaxRecip", 1, ("N", "HALF"), _recip, _recip_signature,
                           doc="a bf16 row's softmax reciprocal on bits: 16 bits (HALF) of RN(1 / s), s the sum from +0 of "
                               "MUFU.EX2.FTZ(RN(RN(x_j - m) * RN32(log2 e))), every NaN 0x7FFFFFFF")
SoftmaxOut = definition("SoftmaxOut", 1, softmax_out, (("x", BF16), ("m", BF16), ("r", F32)), BF16,
                        doc="one softmax output on bits: bf16_rn(RN(MUFU.EX2.FTZ(RN(RN(x - m) * RN32(log2 e))) * r)), NaN 0x7FFF")
EmbeddingCoordinate = specialized("EmbeddingCoordinate", 1, ("V",), _embedding, _embedding_signature,
                                  doc="one coordinate of the embedding plus position on bits: Bf16AddF2fp_v1(col[tok], pos), "
                                      "col[tok] 0x7FFF for a 16-bit tok >= V, NaN 0x7FFF")
RmsSumSqPartHalf = specialized("RmsSumSqPartHalf", 1, ("C", "HALF"),
                               *_f32_half(R.RmsSumSqPart, lambda S: lambda C, *xs: R.sum_squares(C, xs)),
                               doc="16 bits (HALF) of RmsSumSqPart_v1{C} on bits: RN(x_i^2) summed from +0 in ascending order, "
                                   "every NaN 0x7FFFFFFF")
F32SumPartHalf = specialized("F32SumPartHalf", 1, ("G", "HALF"),
                             *_f32_half(R.F32SumPart, lambda S: lambda C, *s: R.f32_sum(C, s)),
                             doc="16 bits (HALF) of F32SumPart_v1{G} on bits: G f32 words summed from +0 in ascending order, "
                                 "every NaN 0x7FFFFFFF")
RmsNormRecipHalf = specialized("RmsNormRecipHalf", 1, ("N", "G", "EPS", "HALF"),
                               *_f32_half(R.RmsNormRecip, lambda S: lambda C, *s: R.recip_root(C, R.f32_sum(C, s), S.N,
                                                                                               R.f32_bits(S.EPS))),
                               doc="16 bits (HALF) of RmsNormRecip_v1{N, G, EPS} on bits: RN(1 / RN(sqrt(RN(RN(acc / N) + "
                                   "EPS)))), acc the G sums summed from +0 in ascending order; every NaN 0x7FFFFFFF")
