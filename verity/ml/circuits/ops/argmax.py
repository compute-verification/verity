"""The greedy argmax of a bf16 logits row as Definitions of at most 16 output bits (a proof block's outputs), from its V-chunks of C
logits:

    ArgmaxChunkBest_v1{C}     (x: bf16[C]) -> bf16: the chunk's first maximum above -inf, -inf when it has none
    ArgmaxChunkIdx_v1{C}      (x: bf16[C]) -> Value<16>: that maximum's index in the chunk, 0 when it has none
    ArgmaxCombine_v1{V, C}    (x0: bf16, best: bf16[n], idx: Value<16>[n]) -> Value<16>: the row's argmax, n = ceil(V / C)

**The function** is `TokenSelect_v1{V}`'s (`verity_vllm`'s greedy sampling, torch.argmax's lowest index on ties): from
(logits[0], 0), take (x_i, i) iff ``Bf16GtStrict_v1(x_i, best)``, so a NaN is never taken, +0 and -0 are equal, the first of
equal maxima stays, a NaN logits[0] is never left and a row with no number above it gives 0. A chunk's pair is that scan from
(-inf, 0) over the chunk; the combine starts from (x0, 0), x0 the row's logits[0], and takes chunk c's (best[c], cC + idx[c])
iff ``Bf16GtStrict_v1(best[c], run)``. The best after chunks 0 .. c-1 is at least every number in them, so it takes chunk c's
first maximum exactly when the scan would. These are `verity_vllm`'s ``TokenSelectChunk_v2{C}`` (its bf16, and its i32 index's
low 16 bits) and ``TokenSelectCombine_v2{V, C}`` (its i32's low 16 bits), the pieces of ``TokenSelect_v4{V}``, one output word
each: a row's argmax is n best proof blocks, n index proof blocks and one combine, each chunk proof block reading its C logits and
the combine 2n + 1 words. The last chunk of a row V does not divide is ``[V - V mod C, V)``, a chunk proof block of its own width.

**The index word** is ``Value<16>``: `SelectU16_v1` and `U16Add_v1` (wrapping) are `scalar`'s ``SelectI32_v1`` and
``I32Add_v1`` at 16 bits, and V <= 2^16 keeps every index in it. Each select keeps the running value as its *then* branch, under
the comparison's `BitNot`, as ``TokenSelectChunkStep_v1``'s does: its bits are the else branch's (the new logit or constant)
XOR one AND each, so the scan lowers to a circuit linear in C.
"""
from __future__ import annotations

from verity.ml.circuits.defs import composite, primitive
from verity.ml.circuits.types import Array, Value

from . import prims as P
from . import scalar as SC
from .prims import BF16

__all__ = ["NEG_INF_BF16", "U16", "ArgmaxChunkBest", "ArgmaxChunkIdx", "ArgmaxCombine", "SelectU16", "U16Add", "chunk_count",
           "refuse"]

U16 = Value(16)
#: bf16 -inf, a chunk's starting best: every number but -inf is strictly above it
NEG_INF_BF16 = 0xFF80
SPAN = 1 << 16
CONFORMANCE = "protocol-defined"


def refuse(b: dict) -> str | None:
    """`bind`'s refusal of the statics bound so far."""
    c, v = b.get("C"), b.get("V")
    if c is not None and (isinstance(c, bool) or not isinstance(c, int) or not 1 <= c <= SPAN):
        return f"C is a chunk's width, an int in [1, 2^16] (its indices are 16-bit words), not {c!r}"
    if v is not None and (isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= SPAN):
        return f"V is the row's width, an int in [1, 2^16] (its argmax is a 16-bit word), not {v!r}"
    if c is not None and v is not None and c > v:
        return f"a row of V = {v} has no chunk of C = {c} logits (1 <= C <= V)"
    return None


def chunk_count(V: int, C: int) -> int:
    """The V-chunks of a row of V logits: ceil(V / C)."""
    return -(-V // C)


@primitive("SelectU16", 1, [("c", SC.BIT), ("a", U16), ("b", U16)], U16, conformance=CONFORMANCE)
def SelectU16(c: int, a: int, b: int) -> int:
    """c ? a : b on 16-bit words."""
    return a if c & 1 else b


@primitive("U16Add", 1, [("a", U16), ("b", U16)], U16, conformance=CONFORMANCE)
def U16Add(a: int, b: int) -> int:
    """Wrapping 16-bit add."""
    return (a + b) & 0xFFFF


def _keep(B, x, best):
    return B.call(SC.BitNot, B.call(SC.Bf16GtStrict, x, best))


def _chunk_signature(ret):
    return lambda S: ((("x", Array(S.C, BF16)),), ret)


@composite("ArgmaxChunkBest", 1, ["C"], _chunk_signature(BF16), conformance=CONFORMANCE, refuse=refuse)
def ArgmaxChunkBest(B, S, x):
    """The chunk's first maximum above -inf, -inf when it has none (the module docstring)."""
    best = B.call(P.const(16, NEG_INF_BF16))
    for i in range(S.C):
        best = B.call(SC.SelectBf16, _keep(B, x[i], best), best, x[i])
    return best


@composite("ArgmaxChunkIdx", 1, ["C"], _chunk_signature(U16), conformance=CONFORMANCE, refuse=refuse)
def ArgmaxChunkIdx(B, S, x):
    """The index in the chunk of its first maximum above -inf, 0 when it has none (the module docstring). Step 0 leaves the
    index 0 whichever way it goes, and the last step's best is read by nothing, so a chunk of one logit is index 0."""
    best, at = B.call(P.const(16, NEG_INF_BF16)), B.call(P.const(16, 0))
    for i in range(S.C if S.C > 1 else 0):
        keep = _keep(B, x[i], best)
        if i:
            at = B.call(SelectU16, keep, at, B.call(P.const(16, i)))
        if i < S.C - 1:
            best = B.call(SC.SelectBf16, keep, best, x[i])
    return at


def _combine_signature(S):
    n = chunk_count(S.V, S.C)
    return (("x0", BF16), ("best", Array(n, BF16)), ("idx", Array(n, U16))), U16


@composite("ArgmaxCombine", 1, ["V", "C"], _combine_signature, conformance=CONFORMANCE, refuse=refuse)
def ArgmaxCombine(B, S, x0, best, idx):
    """The row's argmax from x0 (its logits[0]) and its chunks' (best, idx) (the module docstring)."""
    n = chunk_count(S.V, S.C)
    run, at = x0, B.call(P.const(16, 0))
    for c in range(n):
        keep = _keep(B, best[c], run)
        k = idx[c] if c == 0 else B.call(U16Add, idx[c], B.call(P.const(16, c * S.C)))
        at = B.call(SelectU16, keep, at, k)
        if c < n - 1:
            run = B.call(SC.SelectBf16, keep, run, best[c])
    return at
