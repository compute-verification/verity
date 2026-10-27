"""A head's attention scores and their softmax as one Boolean Definition, one proof block a head writing its probability row:
the attention of the e2e demo's transformer (`sampled_proofs.one_stage.partition`'s size classes: T bf16 outputs, class 16·T rounded
up).

    QkSoftmaxRow_v1{T, D, DOT}   (q: Array<D, Array<16, Value<1>>>, k: Array<T, Array<D, Array<16, Value<1>>>>)
                                 -> Array<T, Array<16, Value<1>>>
    QkSoftmaxRowPad_v1{T, D, DOT}  (q, k as QkSoftmaxRow's, n: Array<16, Value<1>> of kind position)
                                 -> Array<T, Array<16, Value<1>>>
    QkSoftmaxPvPad_v1{T, K, D, DOT}  (q, k as QkSoftmaxRow's, v0..v{T-1}: Array<D, Array<16, Value<1>>>, n as
                                 QkSoftmaxRowPad's) -> Array<D, Array<16, Value<1>>>
    SoftmaxRow_v1{N}             (x: Array<N, Array<16, Value<1>>>) -> Array<N, Array<16, Value<1>>>
    SoftmaxRowPad_v1{N}          (x: Array<N, Array<16, Value<1>>>, n: Array<16, Value<1>>) -> Array<N, Array<16, Value<1>>>

`QkSoftmaxRow` is `SoftmaxRow{N=T}` of the T scores ``x_j = GemmCoordinate_v3{K=D, DOT}(q, k_j)`` (`gemm.Gemm_v3` over the key
rows; `gemm.GEMM_HOPPER` binds the H200's step). Its q row is 16·D bits and each key row as long, so at D = 64 every row it
reads is over C-Flock's 111-byte floor, which a proof block of short rows alone (a softmax row at T <= 55) doesn't reach. A score
scale is folded into q's producer, as a power of two.

`SoftmaxRow` is `SoftmaxRef_v1{N, bf16}`'s dataflow (`verity.ml.definitions.registry.ref_prims`) on the GPU's operations, so a kernel
matches it bit for bit with ``add.rn.f32``, ``mul.rn.f32``, ``div.rn.f32``, ``ex2.approx.ftz.f32`` and ``cvt.rn.bf16.f32``:

- ``m`` is the first greatest x_i (+0 and -0 one value, so the earlier is kept, as `F32MaxRef_v1`'s fold keeps it), or 0x7FFF
  when any x_i is a NaN.
- ``e_j = MUFU.EX2.FTZ(RN(RN(f32(x_j) - f32(m)) * L))``, L = RN32(log2 e) = 0x3FB8AA3B (`MufuEx2Ftz_v2`'s builder), made once
  for each j. The reference's `F32ExpRn_v1` has no Boolean form, and MUFU.EX2 is what an sm_90 kernel computes, read from the
  sm_89 table as the H200 Definitions read it (`verity.ml.definitions.numerics.fa3_model`: assumed identical on sm_90).
- ``s = (((+0 + e_0) + e_1) + ...) + e_{N-1}``, in ascending order, and ``r = RN(1 / s)``.
- ``p_j = bf16_rn(RN(e_j * r))``.

Every f32 operation's NaN is the GPU's 0x7FFFFFFF and every bf16 rounding F2FP's (NaN 0x7FFF), as in `elementwise`'s `_v4`s;
none is flushed but MUFU.EX2's operand. An f32 sum is not associative, so a split of a longer row is a new Definition that
states its order, never a regrouping of this one.

`QkSoftmaxRowPad` is one statement for every key count n <= T: its first n outputs are `QkSoftmaxRow{T=n}(q, k[:n])`'s words
and the other T - n are +0, the row the softmax kernel writes. n is one u16 word of kind position (`ports.In`), so it is
public, as the static T it replaces was. `SoftmaxRowPad` masks each x_j with j >= n three times:

- in the max, x_j is -inf (0xFF80), the least non-NaN key, and the max keeps the earlier of equal keys, so ``m`` is the
  first n's;
- e_j is +0, so ``s`` is the first n's, since adding +0 to a sum from +0 of MUFU.EX2's results (never -0) is exact;
- p_j is +0, which holds even when s is a NaN.

A masked key row is read only by its score, which the mask drops whole, so no word in it (a NaN, an inf) reaches an output;
the replay binds it to a registered zero row. The circuit is total: n = 0 writes T words +0, and n > T is n = T. Its
specification, `qk_softmax_row_pad_reference`, is for 1 <= n <= T only and refuses any other n.

`QkSoftmaxPvPad` is a head's whole attention in one proof block: `gemm.PvRowPad_v2{T, K, D, DOT}` of
`QkSoftmaxRowPad_v1{T, D, DOT}`'s probability row, V's T key rows and the same n, so the probability row is never an output
and is never committed. Each half reads n as its own statement does, so the composition is total too; it refuses what
either half refuses. One DOT serves both products, the step an H200 attention kernel runs QK^T and PV on.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import OR, const_bits, mux
from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.ports import In, Ports
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array

from .. import fp32 as W
from . import gemm as BG
from .elementwise import specialized
from .mufu import mufu_ex2
from .scalar import bf16_key, gt_u

__all__ = ["LOG2E", "NEG_INF", "QkSoftmaxPvPad", "QkSoftmaxRow", "QkSoftmaxRowPad", "SoftmaxRow", "SoftmaxRowPad",
           "qk_softmax_row_pad_reference", "softmax_exp", "softmax_max", "softmax_row", "softmax_row_pad"]

BF16 = Array(16, BL.BIT)
#: the key count's type: one unsigned word
U16 = Array(16, BL.BIT)
#: RN32(log2 e)
LOG2E = 0x3FB8AA3B
ONE_F = 0x3F800000
#: bf16 -inf, a masked score in the max
NEG_INF = 0xFF80
_NAN = fp.F32_NAN_PTX


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


def softmax_row(C, xs, valid=None):
    """``p_j = bf16_rn(RN(e_j * r))`` for each j, ``r = RN(1 / s)`` and ``s`` the e_j summed from +0 in ascending order. With
    ``valid`` (one bit a word), each x_j whose bit is 0 is -inf in the max, its e_j +0 in the sum, and its p_j +0."""
    if valid is not None:
        xs = [mux(C, v, x, const_bits(NEG_INF, 16)) for v, x in zip(valid, xs, strict=True)]
    keep = (lambda j, w: w) if valid is None else (lambda j, w: [C.AND(valid[j], b) for b in w])
    m = softmax_max(C, xs)
    es = [keep(j, softmax_exp(C, x, m)) for j, x in enumerate(xs)]
    s = fp.f32_const(0)
    for e in es:
        s = fp.f32_add(C, s, e, nan_word=_NAN)
    r = fp.f32_div(C, fp.f32_const(ONE_F), s, nan_word=_NAN)
    return [keep(j, fp.f32_to_bf16(C, fp.f32_mul(C, e, r, nan_word=_NAN), W.BF16_NAN_F2FP)) for j, e in enumerate(es)]


def softmax_row_pad(C, xs, n):
    """`softmax_row` of the first n of ``xs``, then +0 words: x_j takes part when j < n, an unsigned compare against the
    whole word n, so n = 0 masks every word and n >= len(xs) none."""
    return softmax_row(C, xs, [gt_u(C, n, const_bits(j, len(n))) for j in range(len(xs))])


def _row(S: Any) -> Array:
    if not isinstance(S.N, int) or S.N < 1:
        raise ValueError(f"N is the row's length, at least 1, not {S.N!r}")
    return Array(S.N, BF16)


SoftmaxRow = specialized("SoftmaxRow", 1, ("N",), lambda S: lambda C, *xs: softmax_row(C, xs),
                         lambda S: ((("x", _row(S)),), _row(S)),
                         doc="a bf16 row's softmax on bits: bf16_rn(RN(e_j * RN(1 / s))), e_j = MUFU.EX2.FTZ(RN(RN(x_j - m) * "
                             "RN32(log2 e))), m the first greatest x_i (0x7FFF on a NaN), s the e_j summed from +0 in order, "
                             "every f32 NaN 0x7FFFFFFF")


SoftmaxRowPad = specialized("SoftmaxRowPad", 1, ("N",), lambda S: lambda C, *xs: softmax_row_pad(C, xs[:-1], xs[-1]),
                            lambda S: ((("x", _row(S)), ("n", U16)), _row(S)),
                            doc="`SoftmaxRow_v1{N}` of x's first n words (n unsigned), then N - n words +0: a word x_j with "
                                "j >= n is -inf in the max, +0 in the sum and +0 out; n = 0 writes N words +0, n > N is n = N")


def _keys(S: Any) -> int:
    if not isinstance(S.T, int) or S.T < 1:
        raise ValueError(f"T is the number of key rows, at least 1, not {S.T!r}")
    return S.T


@composite("QkSoftmaxRow", 1, ["T", "D", "DOT"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(_keys(S), Array(S.D, BF16)))), Array(_keys(S), BF16)),
           conformance="exact-model-tested", refuse=lambda b: BG.Gemm.refuse({"K": b.get("D")}))
def QkSoftmaxRow(B, S, q, k):
    """A head's probability row on bits: `SoftmaxRow_v1{N=T}` of ``GemmCoordinate_v3{K=D, DOT}(q, k_j)`` for j < T."""
    return B.call(bind(SoftmaxRow, N=S.T), B.call(bind(BG.Gemm, K=S.D, N=S.T, DOT=S.DOT), q, k))


@composite("QkSoftmaxRowPad", 1, ["T", "D", "DOT"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(_keys(S), Array(S.D, BF16))), ("n", U16)), Array(_keys(S), BF16)),
           conformance="exact-model-tested", refuse=lambda b: BG.Gemm.refuse({"K": b.get("D")}),
           ports=Ports(params={"n": In(kind="position")}))
def QkSoftmaxRowPad(B, S, q, k, n):
    """A head's probability row at n of its T key rows, on bits: `SoftmaxRowPad_v1{N=T}` of the T scores
    ``GemmCoordinate_v3{K=D, DOT}(q, k_j)`` and n."""
    return B.call(bind(SoftmaxRowPad, N=S.T), B.call(bind(BG.Gemm, K=S.D, N=S.T, DOT=S.DOT), q, k), n)


@composite("QkSoftmaxPvPad", 1, ["T", "K", "D", "DOT"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(_keys(S), Array(S.D, BF16))),
                       *((f"v{t}", Array(S.D, BF16)) for t in range(_keys(S))), ("n", U16)), Array(S.D, BF16)),
           conformance="exact-model-tested",
           refuse=lambda b: QkSoftmaxRowPad.refuse(b) or BG.PvRowPad.refuse(b),
           ports=Ports(params={"n": In(kind="position")}))
def QkSoftmaxPvPad(B, S, q, k, *rest):
    """A head's attention output row at n of its T key rows, on bits: `PvRowPad_v2{T, K, D, DOT}` of
    `QkSoftmaxRowPad_v1{T, D, DOT}(q, k, n)`, the value rows v0..v{T-1} and n."""
    *vs, n = rest
    p = B.call(bind(QkSoftmaxRowPad, T=S.T, D=S.D, DOT=S.DOT), q, k, n)
    return B.call(bind(BG.PvRowPad, T=S.T, K=S.K, D=S.D, DOT=S.DOT), p, *vs, n)


def qk_softmax_row_pad_reference(q, k, n: int, DOT) -> np.ndarray:
    """What `QkSoftmaxRowPad_v1{T, D, DOT}` computes at key count n, by its specification: `QkSoftmaxRow_v1{T=n, D, DOT}` of q
    and k's first n rows, then T - n words +0, for a batch (q ``(B, D)``, k ``(B, T, D)``, bf16 words). Refuses n outside
    1..T, for which the circuit's words (all +0 at n = 0, n = T's above T) are not this function's."""
    from verity.ml.circuits.evaluation.bits import evaluate_words

    q, k = np.asarray(q, dtype=np.uint16), np.asarray(k, dtype=np.uint16)
    T, D = k.shape[1], k.shape[2]
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 1 <= n <= T:
        raise ValueError(f"n is a key count, 1 <= n <= T = {T}, not {n!r}")
    out = np.zeros((len(q), T), dtype=np.uint16)
    out[:, :n] = np.asarray(evaluate_words(bind(QkSoftmaxRow, T=int(n), D=D, DOT=DOT), q, k[:, :n])).reshape(len(q), n)
    return out
