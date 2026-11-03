"""A linear of a circuit cut at its linears, whose product PoUW credits: the circuit holds the linear's A row to PoUW's table
of them and takes its y as a registered value, computing neither.

    LinearARow_v1{K,R,ROUND}   (x: Array<R, Array<K/R, BF16>>) -> Array<K, F32>     its A row: each of x's K elements widened
    LinearOut_v1{N,K,ROUND}    (y: Array<N, BF16>) -> Array<N, BF16>                 y as registered, with no gate
    each _v2                   σ of its _v1 (`verity.ml.definitions.boolean.sigma_lift`)

**The A row** is the FP32 input row Pearl-C's A rows are (`verity.core.protocols.pouw.circuit.pc8`): the call's x widened to
FP32, exact (`Bf16ToF32_v1` on each element, the encoding << 16), x read as R rows of K/R elements in order, so a cut reads
it as its writers hold it (one row after a norm, a head's at o_proj). **y** is the linear's output as its registered value
holds it, which the unit hands on unchanged: its output row is its input row, and it has no other.

ROUND, which no gate reads, is the round of the table a unit reads, and LinearOut's K, which no gate reads either, the
contraction of its linear: each round's readers of each K's tables are templates of their own, so a sample stratified by
template has a reader of every table.
"""
from __future__ import annotations

from verity.ml.circuits.defs import composite
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array
from verity.ml.definitions.boolean import sigma_lift
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.int8_exact import row_parts_refusal
from verity.ml.definitions.registry.prims import BF16, F32

__all__ = ["LinearARow", "LinearARowV2", "LinearOut", "LinearOutV2"]

A_CONFORMANCE = ("protocol-defined; a cut linear's A row, its x's BF16 elements widened to FP32 exactly (Bf16ToF32_v1 each): "
                 "the FP32 input row Pearl-C's A rows are (verity.core.protocols.pouw.circuit.pc8)")
OUT_CONFORMANCE = "protocol-defined; a cut linear's y as its registered value holds it: its one output row is its one input row"


def _round(b: dict) -> str | None:
    if "ROUND" in b and not (type(b["ROUND"]) is int and b["ROUND"] >= 0):
        return f"ROUND={b['ROUND']!r}: a round is a natural number"
    return None


def _a_refusal(b: dict) -> str | None:
    """`bind`'s refusal for `LinearARow_v1`: x is R rows of K/R elements (`int8_exact.row_parts_refusal`), and ROUND a natural
    number."""
    return row_parts_refusal(b) or _round(b)


def _out_refusal(b: dict) -> str | None:
    """`bind`'s refusal for `LinearOut_v1`: N and K positive integers, and ROUND a natural number."""
    for name in ("N", "K"):
        if name in b and not (type(b[name]) is int and b[name] >= 1):
            return f"{name}={b[name]!r}: y's elements and its linear's contraction are positive integers"
    return _round(b)


@composite("LinearARow", 1, ["K", "R", "ROUND"], lambda S: ((("x", Array(S.R, Array(S.K // S.R, BF16))),), Array(S.K, F32)),
           conformance=A_CONFORMANCE, refuse=_a_refusal)
def LinearARow(B, S, x):
    """The A row of x: each of its K elements widened to FP32 (`Bf16ToF32_v1`), x viewed as one array of K by reference.
    ``ROUND``, the round of the A table the row is held to, is read by no gate."""
    return B.batch(P.Bf16ToF32, Coll(Array(S.K, BF16), x.refs), axes=(0,))


@composite("LinearOut", 1, ["N", "K", "ROUND"], lambda S: ((("y", Array(S.N, BF16)),), Array(S.N, BF16)),
           conformance=OUT_CONFORMANCE, refuse=_out_refusal)
def LinearOut(B, S, y):
    """y itself, with no gate. ``K`` and ``ROUND``, the table y is read from, are read by no gate."""
    return y


LinearARowV2 = sigma_lift(LinearARow, 2)
LinearOutV2 = sigma_lift(LinearOut, 2)
