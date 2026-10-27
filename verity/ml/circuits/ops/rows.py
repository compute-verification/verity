"""Rows as composites: one proof block a row, writing the whole row (the e2e demo's decoder).

    Bf16AddRow_v1{N}   (a: Array<N, bf16>, b: Array<N, bf16>) -> Array<N, bf16>    the residual add: Bf16AddF2fp_v1 per element
    Argmax_v1{V}       (logits: Array<V, bf16>) -> Value<16>                      greedy selection: the 16-bit index of the
                                                                                    first maximum, `TokenSelect_v4`'s function

**What sm_90 computes**, which a kernel matches bit for bit:

* `Bf16AddRow_v1{N}`: each element ``bf16_rn(f32(a[j]) + f32(b[j]))``: the words widened to f32 (exact), one ``add.rn.f32``
  with subnormals kept (no ``.ftz``), then ``cvt.rn.bf16.f32`` (SASS ``F2FP.BF16.F32``), round to nearest even, every NaN
  0x7FFF. In CUDA, ``__float2bfloat16_rn(__bfloat162float(a) + __bfloat162float(b))`` compiled without fast math.
* `Argmax_v1{V}`: the scan from (logits[0], 0) that takes (logits[k], k) iff ``logits[k] > best`` as an f32 comparison of
  the widened words (``setp.gt.f32``: false when either is a NaN, +0 equal to -0). So it is 0 when logits[0] is a NaN,
  and otherwise the least index of the greatest non-NaN logit: a parallel reduction matches it when it orders a NaN below
  every number, breaks ties to the lower index, and returns 0 for a NaN logits[0]. This is `TokenSelect_v1{V}`'s function
  (and `TokenSelect_v4`'s, its V-chunked form, in `verity_vllm`'s registry) with the index in 16 bits; ``torch.argmax``
  differs from it on a NaN (it returns the first NaN's index).

`ops.boolean.rows` has both on bits.
"""
from __future__ import annotations

from typing import Callable, Sequence

from verity.ml.circuits.defs import composite
from verity.ml.circuits.types import Array, Value

from . import fp32 as F
from . import prims as P
from . import scalar as SC

__all__ = ["INDEX", "MAX_V", "Argmax", "Bf16AddRow", "add_row", "argmax"]

BF16 = Value(16)
#: the selected index: an unsigned 16-bit word
INDEX = Value(16)
#: the most logits an index of 16 bits selects among
MAX_V = 1 << 16


def _refuse(name: str, lo: int, hi: int | None, what: str) -> Callable[[dict], str | None]:
    """`bind`'s refusal of a static ``name`` that isn't an int in [lo, hi]."""

    def refuse(b: dict) -> str | None:
        n = b.get(name)
        if n is None:
            return None
        if isinstance(n, bool) or not isinstance(n, int) or n < lo or (hi is not None and n > hi):
            return f"{name}={n!r}: {what}"
        return None

    return refuse


refuse_n = _refuse("N", 1, None, "a row has a positive int of elements")
refuse_v = _refuse("V", 2, MAX_V, f"an argmax selects among 2 to {MAX_V} logits, so its index fits 16 bits")


# ---- references ----------------------------------------------------------------------------------------------------------

def add_row(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """`Bf16AddRow_v1{N = len(a)}`'s reference: ``Bf16AddF2fp_v1`` per element."""
    if len(a) != len(b):
        raise ValueError(f"a residual add of rows of {len(a)} and {len(b)} words")
    return [F.Bf16AddF2fp.evaluate(x, y) for x, y in zip(a, b)]


def argmax(logits: Sequence[int]) -> int:
    """`Argmax_v1{V = len(logits)}`'s reference: the scan from (logits[0], 0), (logits[k], k) taken iff
    ``Bf16GtStrict_v1(logits[k], best)``."""
    best, at = logits[0], 0
    for k in range(1, len(logits)):
        if SC.Bf16GtStrict.evaluate(logits[k], best):
            best, at = logits[k], k
    return at


# ---- the Definitions -----------------------------------------------------------------------------------------------------

@composite("Bf16AddRow", 1, ["N"], lambda S: ((("a", Array(S.N, BF16)), ("b", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested", refuse=refuse_n)
def Bf16AddRow(B, S, a, b):
    """The residual add of a row of N bf16 words, one proof block a row: `Bf16AddF2fp_v1` per element, ``bf16_rn(f32(a) + f32(b))``
    with every NaN 0x7FFF (``add.rn.f32`` then ``cvt.rn.bf16.f32`` on sm_90)."""
    return B.batch(F.Bf16AddF2fp, a, b, axes=(0, 0))


@composite("Argmax", 1, ["V"], lambda S: ((("logits", Array(S.V, BF16)),), INDEX), conformance="exact-model-tested",
           refuse=refuse_v)
def Argmax(B, S, logits):
    """Greedy selection over V bf16 logits, one proof block a row: the scan from (logits[0], 0) taking (logits[k], k) iff
    `Bf16GtStrict_v1(logits[k], best)`, so the first maximum wins, a NaN is never taken and a NaN logits[0] is never left;
    the index as an unsigned 16-bit word (`TokenSelect_v1{V}`'s function)."""
    best, at = logits[0], B.call(P.const(16, 0))
    for k in range(1, S.V):
        gt = B.call(SC.Bf16GtStrict, logits[k], best)
        at = B.call(SC.SelectBf16, gt, B.call(P.const(16, k)), at)
        if k < S.V - 1:
            best = B.call(SC.SelectBf16, gt, logits[k], best)
    return at
