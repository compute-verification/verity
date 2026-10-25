"""The per-row s8 quantizer of a W8A8 linear's activations, which makes `gemm.ScaledMmS8Coordinate_v1`'s ``x`` and ``sx`` from the
previous linear's bf16 row, with the ReLU between two linears folded in, as Definitions of at most 16 output bits (a proof
unit's outputs):

    RowAbsMax_v1{N, RELU}    (x: bf16[N]) -> bf16: max |x_k| over the row, relu'd first under RELU
    QuantS8Pair_v1{RELU}     (x: bf16[2], amax: bf16) -> s8[2]: the codes of x[0] and x[1] under amax's scale
    QuantS8Scale_v1{HALF}    (amax: bf16) -> Value<16>: amax's f32 scale, its low (HALF 0) or high (HALF 1) 16 bits

**The function** is the torch loop's ``quantize`` (`verity_torch_loop.loop`) of its row, relu'd first under ``RELU``:

* ReLU on bf16, ``x < 0 ? +0 : x`` (``torch.relu``: -0 and a NaN are themselves), then the exact widening to f32;
* amax = max |x_k| over the row, NaN when any |x_k| is (``amax``'s propagation): the integer max of the f32 encodings with their
  sign bits cleared, whose unsigned order is the order of the values with every NaN above +inf. A max of bf16 values is one of
  them, so amax is a bf16 word (sign 0), and the max of a row's partial maxima is the row's: ``RowAbsMax_v1{m, false}`` over
  ``RowAbsMax_v1{N/m, RELU}`` of the row's m parts is ``RowAbsMax_v1{N, RELU}``;
* scale = RN(amax / 127) (`fp32.F32Div_v1`) when amax > 0 (+inf included), else 1.0 (``torch.where(amax > 0, amax / 127, 1)``:
  a zero, negative or NaN amax);
* code_k = `scalar.F32ToS8RniSat_v1`(RN(x_k / scale)): ``round(x_k / scale).clamp(-128, 127).to(int8)``, ties to even, NaN 0.

A subnormal amax can give a subnormal or zero scale, and an infinite one an infinite scale; both are torch's (x / 0 is ±inf, so
127 or -128, and 0 / 0 or inf / inf NaN, so 0), and the bf16 outputs of a finite linear never reach them.

**The units.** A row's quantization is its amax (one unit, or a tree of them over the row's parts), then H / 2 pair units and
the two scale units, each reading amax: the pair unit of codes 2j and 2j + 1 reads x[2j], x[2j + 1] and amax. In the torch loop
``RELU`` is true on the up_proj -> down_proj edge and false on down_proj -> the next step's up_proj, which has no ReLU; the pair
units of a row take the same ``RELU`` as its amax.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits.defs import composite
from verity.ml.circuits.types import Array

from . import fp32 as F
from . import prims as P
from . import scalar as SC
from .prims import BF16, S8

__all__ = ["F32_127", "F32_INF", "F32_ONE", "QuantS8Pair", "QuantS8Scale", "RowAbsMax", "refuse"]

F32_ONE, F32_127, F32_INF = 0x3F800000, 0x42FE0000, 0x7F800000
CONFORMANCE = "exact-model-tested; the torch loop's quantize on CPU torch (integrations/torch_loop/tests/test_quant.py)"


def refuse(b: dict) -> str | None:
    """`bind`'s refusal of the statics bound so far."""
    n, relu, half = b.get("N", 1), b.get("RELU", False), b.get("HALF", 0)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        return f"N is the row's width, an int >= 1, not {n!r}"
    if not isinstance(relu, bool):
        return f"RELU is a bool (whether the row is relu'd before it is quantized), not {relu!r}"
    if isinstance(half, bool) or half not in (0, 1):
        return f"HALF is 0 (the scale's low 16 bits) or 1 (its high 16 bits), not {half!r}"
    return None


def _relu(B: Any, S: Any, x: Any) -> Any:
    if not S.RELU:
        return x
    zero = B.call(P.const(16, 0))
    return B.batch(SC.SelectBf16, B.batch(SC.Bf16GtStrict, zero, x, axes=(None, 0)), zero, x, axes=(0, None, 0))


def _scale(B: Any, amax: Any) -> Any:
    """amax's f32 scale: RN(amax / 127) when amax > 0, else 1.0."""
    a = B.call(P.Bf16ToF32, amax)
    pos = B.call(SC.BitAnd, B.call(SC.I32Le, B.call(P.const(32, 1)), a), B.call(SC.I32Le, a, B.call(P.const(32, F32_INF))))
    return B.call(SC.SelectF32, pos, B.call(F.F32Div, a, B.call(P.const(32, F32_127))), B.call(P.const(32, F32_ONE)))


@composite("RowAbsMax", 1, ["N", "RELU"], lambda S: ((("x", Array(S.N, BF16)),), BF16), conformance=CONFORMANCE, refuse=refuse)
def RowAbsMax(B, S, x):
    """max |x_k| over the row, relu'd first under RELU, as a bf16 word (the module docstring)."""
    a = B.batch(SC.F32Fabs, B.batch(P.Bf16ToF32, _relu(B, S, x), axes=(0,)), axes=(0,))
    amax = a[0]
    for k in range(1, S.N):
        amax = B.call(SC.SelectI32, B.call(SC.I32Le, amax, a[k]), a[k], amax)
    return B.call(SC.U32Hi16, amax)


@composite("QuantS8Pair", 1, ["RELU"], lambda S: ((("x", Array(2, BF16)), ("amax", BF16)), Array(2, S8)),
           conformance=CONFORMANCE, refuse=refuse)
def QuantS8Pair(B, S, x, amax):
    """The s8 codes of x[0] and x[1], relu'd first under RELU, under amax's scale (the module docstring)."""
    f = B.batch(P.Bf16ToF32, _relu(B, S, x), axes=(0,))
    return B.batch(SC.F32ToS8RniSat, B.batch(F.F32Div, f, _scale(B, amax), axes=(0, None)), axes=(0,))


@composite("QuantS8Scale", 1, ["HALF"], lambda S: ((("amax", BF16),), SC.U16), conformance=CONFORMANCE, refuse=refuse)
def QuantS8Scale(B, S, amax):
    """amax's f32 scale, its low (HALF 0) or high (HALF 1) 16 bits (the module docstring)."""
    return B.call(SC.U32Hi16 if S.HALF else SC.U32Lo16, _scale(B, amax))
