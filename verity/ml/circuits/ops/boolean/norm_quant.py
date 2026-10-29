"""RMSNorm into the s8 quantizer as Boolean Definitions whose norm output is never committed: a proof block reads a
C-word part of the row x, the gain w's same words and the row's S chunk sums, recomputes the norm's output there and returns
what the quantizer's block over that output returns.

    RmsNormAbsMax_v1{C, S, G, N, EPS}       (x: Array<C, BF16>, w: Array<C, BF16>, s: Array<S, F32>) -> BF16
    RmsNormQuantCodes_v1{C, S, G, N, EPS}   (x: Array<C, BF16>, w: Array<C, BF16>, s: Array<S, F32>, amax: BF16)
                                            -> Array<C, Array<8, Value<1>>>

(BF16 is ``Array<16, Value<1>>``, F32 ``Array<32, Value<1>>``.) Each is its unfused chain with the table between them gone:

- ``RmsNormAbsMax{C, S, G, N, EPS}(x, w, s)`` is ``RowAbsMax_v2{C, false}(RmsNormOutSums_v1{C, S, G, N, EPS}(x, w, s))``, the
  amax of the part's output; ``RowAbsMax_v2{m, false}`` of a row's m part amaxes is the row's (`ops.quant`);
- ``RmsNormQuantCodes{C, S, G, N, EPS}(x, w, s, amax)`` is ``QuantChunkS8_v2{C, false}(RmsNormOutSums_v1{C, S, G, N, EPS}(x, w,
  s), amax)``'s codes, the part's codes under the row's amax; the scale, the chunk's other output, is ``QuantScaleS8_v1(amax)``.

The body is `rmsnorm`'s four-part composition (``RmsNormOutSums``'s statement): ``F32SumPart_v1{S / G}`` a group of the
chunk sums, ``RmsNormRecip_v1{N, G, EPS}`` of the G group sums once, then ``RmsNormOut_v1{1}`` a word; and the quantizer's
steps, ``RowAbsMax_v2{C, false}`` of the output, or ``QuantScaleS8_v1`` of amax once and ``QuantCodeS8_v1`` a word. Every
callee is one small circuit traced once and called a word at a time. Flock's stager holds 4-7 KB a gate of such a block (the
amax at C = 2,048 peaks at 19.2 GB, the codes at C = 512 at 17.8 GB), so a frozen-width row is parts of these, never one block.
A bf16 multiply and an f32 add are no MACs: nothing either recomputes is a linear's.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.types import Array

from . import quant as BQ
from . import rmsnorm as RN

__all__ = ["RmsNormAbsMax", "RmsNormQuantCodes"]

BF16, F32, CODE = RN.BF16, RN.F32, BQ.CODE
STATICS = ["C", "S", "G", "N", "EPS"]


def _ports(S: Any) -> tuple[tuple[str, Any], ...]:
    """(x, w, s), refusing what `RmsNormOutSums_v1`'s signature refuses."""
    RN.RmsNormOutSums.signature(S)
    part = Array(S.C, BF16)
    return ("x", part), ("w", part), ("s", Array(S.S, F32))


def _out(B: Any, S: Any, x: Any, w: Any, s: Any) -> Any:
    """`RmsNormOutSums_v1{C, S, G, N, EPS}(x, w, s)`, as its four parts: the C output words."""
    k = S.S // S.G
    groups = B.batch(bind(RN.F32SumPart, G=k), s.reshape(Array(S.G, Array(k, F32))), axes=(0,))
    rs = B.call(bind(RN.RmsNormRecip, N=S.N, G=S.G, EPS=S.EPS), groups)
    one = Array(S.C, Array(1, BF16))
    out = B.batch(bind(RN.RmsNormOut, C=1), x.reshape(one), rs, w.reshape(one), axes=(0, None, 0))
    return out.reshape(Array(S.C, BF16))


@composite("RmsNormAbsMax", 1, STATICS, lambda S: (_ports(S), BF16), conformance="exact-model-tested")
def RmsNormAbsMax(B, S, x, w, s):
    """`RowAbsMax_v2{C, false}` of `RmsNormOutSums_v1{C, S, G, N, EPS}(x, w, s)` (the module docstring)."""
    return B.call(bind(BQ.RowAbsMax, N=S.C, RELU=False), _out(B, S, x, w, s))


@composite("RmsNormQuantCodes", 1, STATICS, lambda S: ((*_ports(S), ("amax", BF16)), Array(S.C, CODE)),
           conformance="exact-model-tested")
def RmsNormQuantCodes(B, S, x, w, s, amax):
    """`QuantChunkS8_v2{C, false}`'s codes of `RmsNormOutSums_v1{C, S, G, N, EPS}(x, w, s)` under amax (the module
    docstring)."""
    return B.batch(BQ.QuantCodeS8, _out(B, S, x, w, s), B.call(BQ.QuantScaleS8, amax), axes=(0, None))
