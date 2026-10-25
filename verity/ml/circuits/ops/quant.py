"""The per-row s8 quantizer of a W8A8 linear's activations, which makes `gemm.ScaledMmS8Coordinate_v1`'s ``x`` and ``sx`` from the
previous linear's bf16 row, with the ReLU between two linears folded in, as Definitions of at most 16 output bits (a proof
unit's outputs):

    QuantS8Pair_v1{H, RELU}          (x: bf16[H]) -> s8[2]: the codes of x[0] and x[1]
    QuantS8Scale_v1{H, HALF, RELU}   (x: bf16[H]) -> Value<16>: the row's f32 scale, its low (HALF 0) or high (HALF 1) 16 bits

**The function** is the torch loop's ``quantize`` (`verity_torch_loop.loop`) of its row, relu'd first under ``RELU``:

* ReLU on bf16, ``x < 0 ? +0 : x`` (``torch.relu``: -0 and a NaN are themselves), then the exact widening to f32;
* amax = max |x_k| over the row, NaN when any |x_k| is (``amax``'s propagation): the integer max of the f32 encodings with their
  sign bits cleared, whose unsigned order is the order of the values with every NaN above +inf;
* scale = RN(amax / 127) (`fp32.F32Div_v1`) when 0 < amax (+inf included), else 1.0 (``torch.where(amax > 0, amax / 127, 1)``:
  a zero or NaN amax);
* code_k = `scalar.F32ToS8RniSat_v1`(RN(x_k / scale)): ``round(x_k / scale).clamp(-128, 127).to(int8)``, ties to even, NaN 0.

A subnormal amax can give a subnormal or zero scale, and an infinite one an infinite scale; both are torch's (x / 0 is ±inf, so
127 or -128, and 0 / 0 or inf / inf NaN, so 0), and the bf16 outputs of a finite linear never reach them.

**The units.** A row's quantization is H / 2 pair units and the two scale units, each reading the whole row. A pair unit's codes
are those of x[0] and x[1], and its function is symmetric in x[2:], so the unit of codes 2j and 2j + 1 reads the row rotated by
2j and every pair unit of a row is one template. In the torch loop ``RELU`` is true on the up_proj -> down_proj edge and false
on down_proj -> the next step's up_proj, which has no ReLU.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits.defs import composite
from verity.ml.circuits.types import Array

from . import fp32 as F
from . import prims as P
from . import scalar as SC
from .prims import BF16, S8

__all__ = ["F32_127", "F32_INF", "F32_ONE", "QuantS8Pair", "QuantS8Scale", "refuse"]

F32_ONE, F32_127, F32_INF = 0x3F800000, 0x42FE0000, 0x7F800000


def refuse(b: dict) -> str | None:
    """`bind`'s refusal of the statics bound so far."""
    H, relu, half = b.get("H", 2), b.get("RELU", False), b.get("HALF", 0)
    if isinstance(H, bool) or not isinstance(H, int) or H < 2:
        return f"H is the row's width, an int >= 2 (a pair unit reads x[0] and x[1]), not {H!r}"
    if not isinstance(relu, bool):
        return f"RELU is a bool (whether the row is relu'd before it is quantized), not {relu!r}"
    if isinstance(half, bool) or half not in (0, 1):
        return f"HALF is 0 (the scale's low 16 bits) or 1 (its high 16 bits), not {half!r}"
    return None


def _row(B: Any, S: Any, x: Any) -> tuple[Any, Any]:
    """The row as f32, relu'd on bf16 under ``RELU``, and its scale."""
    if S.RELU:
        zero = B.call(P.const(16, 0))
        x = B.batch(SC.SelectBf16, B.batch(SC.Bf16GtStrict, zero, x, axes=(None, 0)), zero, x, axes=(0, None, 0))
    f = B.batch(P.Bf16ToF32, x, axes=(0,))
    a = B.batch(SC.F32Fabs, f, axes=(0,))
    amax = a[0]
    for k in range(1, S.H):
        amax = B.call(SC.SelectI32, B.call(SC.I32Le, amax, a[k]), a[k], amax)
    pos = B.call(SC.BitAnd, B.call(SC.I32Le, B.call(P.const(32, 1)), amax), B.call(SC.I32Le, amax, B.call(P.const(32, F32_INF))))
    scale = B.call(SC.SelectF32, pos, B.call(F.F32Div, amax, B.call(P.const(32, F32_127))), B.call(P.const(32, F32_ONE)))
    return f, scale


@composite("QuantS8Pair", 1, ["H", "RELU"], lambda S: ((("x", Array(S.H, BF16)),), Array(2, S8)),
           conformance="exact-model-tested; the torch loop's quantize on CPU torch (integrations/torch_loop/tests/test_quant.py)",
           refuse=refuse)
def QuantS8Pair(B, S, x):
    """The s8 codes of x[0] and x[1] under the row's scale (the module docstring)."""
    f, scale = _row(B, S, x)
    return B.batch(SC.F32ToS8RniSat, B.batch(F.F32Div, f[:2], scale, axes=(0, None)), axes=(0,))


@composite("QuantS8Scale", 1, ["H", "HALF", "RELU"], lambda S: ((("x", Array(S.H, BF16)),), SC.U16),
           conformance="exact-model-tested; the torch loop's quantize on CPU torch (integrations/torch_loop/tests/test_quant.py)",
           refuse=refuse)
def QuantS8Scale(B, S, x):
    """The row's f32 scale, its low (HALF 0) or high (HALF 1) 16 bits (the module docstring)."""
    _, scale = _row(B, S, x)
    return B.call(SC.U32Hi16 if S.HALF else SC.U32Lo16, scale)
