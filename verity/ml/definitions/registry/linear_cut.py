"""A linear of a circuit cut at its linears, whose product PoUW credits: the circuit holds the linear's A row to the rows
PoUW's call of it commits and takes its y as that call commits it, computing neither.

    LinearARow_v1{K,R}   (x: Array<R, Array<K/R, BF16>>) -> Array<K, F32>   its A row: each of x's K elements widened
    LinearOut_v1{N,W}    (y: Array<N, F32>) -> N/W × Array<W, BF16>         y: each committed word's high half, no gate
    each _v2             σ of its _v1 (`verity.ml.definitions.boolean.sigma_lift`)

**The A row** is the FP32 input row Pearl-C's A rows are (`verity.core.protocols.pouw.circuit.pc8`): the call's x widened to
FP32, exact (`Bf16ToF32_v1` on each element, the encoding << 16), x read as R rows of K/R elements in order, so a cut reads
it as its writers hold it (one row after a norm, a head's at o_proj). **y** is read as the call commits it, a row of its N
BF16 words each widened to FP32 exactly as its A rows are, and the unit hands on each word's high half, its BF16 encoding
(`U32Hi16_v1`, a wire), as N / W rows of W words, so each reader of y reads whole rows of it (a head's at q/k/v): its
output rows are its input row's high halves in order, and it has no gate.

A Definition is its shape alone: which call's rows a unit reads is the circuit's registered read of it, no parameter of
the Definition, so every cut linear of one shape is a unit of one template, whatever its round and its call.
"""
from __future__ import annotations

from verity.ml.circuits.defs import composite
from verity.ml.circuits.ops.scalar import U32Hi16
from verity.ml.circuits.refs import Coll, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.boolean import sigma_lift
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.int8_exact import row_parts_refusal
from verity.ml.definitions.registry.prims import BF16, F32

__all__ = ["LinearARow", "LinearARowV2", "LinearOut", "LinearOutV2"]

A_CONFORMANCE = ("protocol-defined; a cut linear's A row, its x's BF16 elements widened to FP32 exactly (Bf16ToF32_v1 each): "
                 "the FP32 input row Pearl-C's A rows are (verity.core.protocols.pouw.circuit.pc8)")
OUT_CONFORMANCE = ("protocol-defined; a cut linear's y as its call commits it, each BF16 word widened to FP32: its N / W "
                   "output rows are its one input row's high halves (U32Hi16_v1 each) in order")


def _out_refusal(b: dict) -> str | None:
    """`bind`'s refusal for `LinearOut_v1`: y's N words are handed on in rows of W, so N and W are positive integers and W
    divides N."""
    N, W = b.get("N"), b.get("W")
    for name, v in (("N", N), ("W", W)):
        if v is not None and (type(v) is not int or v < 1):
            return f"{name}={v!r}: y's N words are handed on in rows of W, so N and W are positive integers"
    if N is not None and W is not None and N % W:
        return f"N={N}, W={W}: y's N words are handed on in rows of W, so W divides N"
    return None


@composite("LinearARow", 1, ["K", "R"], lambda S: ((("x", Array(S.R, Array(S.K // S.R, BF16))),), Array(S.K, F32)),
           conformance=A_CONFORMANCE, refuse=row_parts_refusal)
def LinearARow(B, S, x):
    """The A row of x: each of its K elements widened to FP32 (`Bf16ToF32_v1`), x viewed as one array of K by reference."""
    return B.batch(P.Bf16ToF32, Coll(Array(S.K, BF16), x.refs), axes=(0,))


@composite("LinearOut", 1, ["N", "W"], lambda S: ((("y", Array(S.N, F32)),), Tuple(*[Array(S.W, BF16)] * (S.N // S.W))),
           conformance=OUT_CONFORMANCE, refuse=_out_refusal)
def LinearOut(B, S, y):
    """y's BF16 encodings, each committed word's high half (`U32Hi16_v1`, a wire), as N / W rows of W in order."""
    h = B.batch(U32Hi16, y, axes=(0,))
    return tuple_of(*(h[k * S.W:(k + 1) * S.W] for k in range(S.N // S.W)))


LinearARowV2 = sigma_lift(LinearARow, 2)
LinearOutV2 = sigma_lift(LinearOut, 2)
