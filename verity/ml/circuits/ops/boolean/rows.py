"""Row proof blocks on bits: the next version of each `ops.rows` word Definition, each one proof block a row that writes the whole
row (the e2e demo's decoder).

    Bf16AddRow_v2{N}   (a, b: Array<N, bf16 bits>) -> Array<N, bf16 bits>          word view: Bf16AddRow_v1{N}
    Argmax_v2{V}       (logits: Array<V, bf16 bits>) -> Array<16, Value<1>>        word view: Argmax_v1{V}
        bf16 bits = Array<16, Value<1>>; N >= 1, 2 <= V <= 2^16

`Bf16AddRow_v2` is `Bf16AddF2fp_v2` (`elementwise`) per element. `Argmax_v2` is traced (`elementwise.specialized`): each
logit's order key and NaN bit (`scalar.bf16_key`) once, the running best held as its key and NaN bit, and each step's select
with the logit and the constant index as its else branch, so a step costs an AND a key bit, an AND a live index bit and the
comparison, and the forms stay flat.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, OR, ZERO, const_bits, mux
from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.types import Array

from .. import rows as W
from . import elementwise as EW
from .scalar import bf16_key, gt_u

__all__ = ["Argmax", "Bf16AddRow"]

BIT = BL.BIT
BF16 = Array(16, BIT)
INDEX = Array(16, BIT)


@composite("Bf16AddRow", 2, ["N"], lambda S: ((("a", Array(S.N, BF16)), ("b", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested", refuse=W.refuse_n)
def Bf16AddRow(B, S, a, b):
    """`Bf16AddRow_v1{N}`'s body on bits: `Bf16AddF2fp_v2` per element."""
    return B.batch(EW.Bf16AddF2fp, a, b, axes=(0, 0))


def _argmax(S):
    width = (S.V - 1).bit_length()

    def build(C, *logits):
        keys = [bf16_key(C, x) for x in logits]
        nan, key = keys[0]
        at = const_bits(0, width)
        for k in range(1, S.V):
            nk, kk = keys[k]
            keep = NOT(C.AND(NOT(OR(C, nan, nk)), gt_u(C, kk, key)))
            at = mux(C, keep, at, const_bits(k, width))
            if k < S.V - 1:
                nan, key = mux(C, keep, [nan], [nk])[0], mux(C, keep, key, kk)
        return at + [ZERO] * (16 - width)

    return build


Argmax = EW.specialized(
    "Argmax", 2, ("V",), _argmax, lambda S: ((("logits", Array(S.V, BF16)),), INDEX),
    word=lambda b: bind(W.Argmax, V=b["V"]),
    doc="Argmax_v1{V} on bits: the scan from (logits[0], 0) taking (logits[k], k) iff Bf16GtStrict(logits[k], best), on the "
        "logits' order keys, the index as 16 bits")
Argmax.refuse = W.refuse_v
Bf16AddRow.word = lambda b: bind(W.Bf16AddRow, N=b["N"])  # type: ignore[attr-defined]
