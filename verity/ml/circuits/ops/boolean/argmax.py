"""The greedy argmax's proof blocks on bits: the next version of each word Definition of `ops.argmax`, the same function.

    ArgmaxChunkBest_v2{C}     (x: Array<C, Array<16, Value<1>>>) -> Array<16, Value<1>>        word view: ArgmaxChunkBest_v1{C}
    ArgmaxChunkIdx_v2{C}      (x: Array<C, Array<16, Value<1>>>) -> Array<16, Value<1>>        word view: ArgmaxChunkIdx_v1{C}
    ArgmaxCombine_v2{V, C}    (x0: Array<16, Value<1>>, best, idx: Array<n, Array<16, Value<1>>>) -> Array<16, Value<1>>
                                                                                             word view: ArgmaxCombine_v1{V, C}
    SelectU16_v2              (c: Value<1>, a, b: Array<16, Value<1>>) -> Array<16, Value<1>>  word view: SelectU16_v1
    U16Add_v2                 (a, b: Array<16, Value<1>>) -> Array<16, Value<1>>               word view: U16Add_v1

A chunk's two are one scan traced at its C (`elementwise.specialized`), each keeping only the gates its output reads. The scan
carries the running best's key (`scalar.bf16_key`), so a step keys only its logit and compares the two keys (`scalar.gt_u`). The
best is never a NaN, since it starts at -inf and takes only a logit above it, so a step takes x_i iff x_i is no NaN and its key is
the greater: ``Bf16GtStrict_v1(x_i, best)``. The combine compares by `scalar.bf16_gt_strict`, since x0 may be a NaN, and adds
each chunk's offset cC by `forms.add`, a constant folded into its gates. Each select is `forms.mux` with the running value its
then branch, as the word Definitions have it.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, add, const_bits, mux
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array

from .. import argmax as A
from .elementwise import specialized
from .scalar import bf16_gt_strict, bf16_key, gt_u, select

__all__ = ["ArgmaxChunkBest", "ArgmaxChunkIdx", "ArgmaxCombine", "SelectU16", "U16Add"]

BIT = BL.BIT
WORD = Array(16, BIT)


def _scan(C, xs):
    """A chunk's (best, idx) on bits: from (-inf, 0), x_i taken iff it is no NaN and its key is above the best's."""
    best, at = const_bits(A.NEG_INF_BF16, 16), const_bits(0, 16)
    key = bf16_key(C, best)[1]
    for i, x in enumerate(xs):
        nan, k = bf16_key(C, x)
        keep = NOT(C.AND(NOT(nan), gt_u(C, k, key)))
        best, key, at = mux(C, keep, best, list(x)), mux(C, keep, key, k), mux(C, keep, at, const_bits(i, 16))
    return best, at


def _combine(S):
    n = A.chunk_count(S.V, S.C)

    def build(C, x0, *parts):
        best, idx = parts[:n], parts[n:]
        run, at = list(x0), const_bits(0, 16)
        for c in range(n):
            keep = NOT(bf16_gt_strict(C, best[c], run)[0])
            k = list(idx[c]) if c == 0 else add(C, idx[c], const_bits(c * S.C, 16), width=16)
            at = mux(C, keep, at, k)
            run = mux(C, keep, run, list(best[c]))
        return at
    return build


def _chunk_signature(S):
    return (("x", Array(S.C, WORD)),), WORD


def _combine_signature(S):
    n = A.chunk_count(S.V, S.C)
    return (("x0", WORD), ("best", Array(n, WORD)), ("idx", Array(n, WORD))), WORD


SelectU16 = definition("SelectU16", 2, lambda C, c, a, b: select(C, c[0], a, b), (("c", BIT), ("a", WORD), ("b", WORD)), WORD,
                       word=A.SelectU16, doc="c ? a : b on 16-bit words, on bits: SelectU16_v1")
U16Add = definition("U16Add", 2, lambda C, a, b: add(C, a, b, width=16), (("a", WORD), ("b", WORD)), WORD, word=A.U16Add,
                    doc="wrapping 16-bit add on bits: U16Add_v1, a ripple-carry adder")
ArgmaxChunkBest = specialized("ArgmaxChunkBest", 2, ("C",), lambda S: lambda C, *x: _scan(C, x)[0], _chunk_signature,
                              doc="`ArgmaxChunkBest_v1`'s chunk maximum on bits, one circuit from its scan",
                              word=lambda b: bind(A.ArgmaxChunkBest, **b))
ArgmaxChunkIdx = specialized("ArgmaxChunkIdx", 2, ("C",), lambda S: lambda C, *x: _scan(C, x)[1], _chunk_signature,
                             doc="`ArgmaxChunkIdx_v1`'s index of the chunk maximum on bits, one circuit from its scan",
                             word=lambda b: bind(A.ArgmaxChunkIdx, **b))
ArgmaxCombine = specialized("ArgmaxCombine", 2, ("V", "C"), _combine, _combine_signature,
                            doc="`ArgmaxCombine_v1`'s row argmax from its chunks on bits, one circuit from its builder",
                            word=lambda b: bind(A.ArgmaxCombine, **b))
ArgmaxChunkBest.refuse = ArgmaxChunkIdx.refuse = ArgmaxCombine.refuse = A.refuse  # type: ignore[attr-defined]
