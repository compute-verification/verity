"""The per-row s8 quantizer of a W8A8 linear's activations, which makes `gemm.ScaledMmS8Coordinate_v1`'s ``x`` and ``sx`` from the
previous linear's bf16 row, with the ReLU between two linears folded in, as row proof blocks under the size classes (a proof
block's outputs fit a public power of two from 16 to 262,144 bits):

    QuantRowS8_v1{N, RELU}      (x: bf16[N]) -> (s8[N], f32): the row's codes and its scale, N * 8 + 32 output bits
    RowAbsMax_v1{N, RELU}       (x: bf16[N]) -> bf16: max |x_k| over the row, relu'd first under RELU
    QuantChunkS8_v1{N, RELU}    (x: bf16[N], amax: bf16) -> (s8[N], f32): a chunk's codes under the row's amax, and the scale

**The function** is torch's per-row s8 quantize of its row, relu'd first under ``RELU``:

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

**The proof blocks.** A row is one ``QuantRowS8_v1`` proof block while its N * 8 + 32 output bits fit the widest class
(N <= 32,764) and the proof block fits the proof system's block. A wider row is ``RowAbsMax_v1`` over the whole row, then one
``QuantChunkS8_v1`` a chunk, each reading the chunk and that amax; chunk k's codes are the row's codes k * N .. k * N + N - 1, and
every chunk's scale is the row's. The amax can be taken in parts: a part's amax is its own ReLU and magnitude, so
``RowAbsMax_v1{m + j, RELU}`` over the last part's m values and the other j parts' amaxes is the row's (a combine that reads a
part's values, for a proof system that refuses a proof block all of whose rows are short). ``RELU`` is true on a ReLU MLP's
up_proj -> down_proj edge and false on down_proj -> the next step's up_proj, which has no ReLU; the proof blocks of a row take
the same ``RELU``.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits.defs import composite
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple

from . import fp32 as F
from . import prims as P
from . import scalar as SC
from .prims import BF16, F32, S8

__all__ = ["F32_127", "F32_INF", "F32_ONE", "QuantChunkS8", "QuantRowS8", "RowAbsMax", "refuse"]

F32_ONE, F32_127, F32_INF = 0x3F800000, 0x42FE0000, 0x7F800000
CONFORMANCE = "exact-model-tested; numpy float32, the same IEEE operations as torch's quantize (verity/ml/circuits/ops/tests/test_quant.py)"


def refuse(b: dict) -> str | None:
    """`bind`'s refusal of the statics bound so far."""
    n, relu = b.get("N", 1), b.get("RELU", False)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        return f"N is the row's width, an int >= 1, not {n!r}"
    if not isinstance(relu, bool):
        return f"RELU is a bool (whether the row is relu'd before it is quantized), not {relu!r}"
    return None


def _widened(B: Any, S: Any, x: Any) -> Any:
    """The row relu'd under RELU, widened to f32."""
    if S.RELU:
        zero = B.call(P.const(16, 0))
        x = B.batch(SC.SelectBf16, B.batch(SC.Bf16GtStrict, zero, x, axes=(None, 0)), zero, x, axes=(0, None, 0))
    return B.batch(P.Bf16ToF32, x, axes=(0,))


def _amax(B: Any, S: Any, f: Any) -> Any:
    """max |f_k| as an f32 word: the unsigned max of the magnitudes' encodings, pairwise, so that each word is a max of at most
    log2(N) selections (a chain's k-th would carry k into the Boolean lowering's linear forms)."""
    a = B.batch(SC.F32Fabs, f, axes=(0,))
    level = [a[k] for k in range(S.N)]
    while len(level) > 1:
        odd = level[len(level) & ~1:]
        level = [B.call(SC.SelectI32, B.call(SC.I32Le, p, q), q, p) for p, q in zip(level[0::2], level[1::2])] + odd
    return level[0]


def _scale(B: Any, a: Any) -> Any:
    """The f32 amax ``a``'s scale: RN(a / 127) when a > 0, else 1.0."""
    pos = B.call(SC.BitAnd, B.call(SC.I32Le, B.call(P.const(32, 1)), a), B.call(SC.I32Le, a, B.call(P.const(32, F32_INF))))
    return B.call(SC.SelectF32, pos, B.call(F.F32Div, a, B.call(P.const(32, F32_127))), B.call(P.const(32, F32_ONE)))


def _codes(B: Any, f: Any, scale: Any) -> Any:
    return tuple_of(B.batch(SC.F32ToS8RniSat, B.batch(F.F32Div, f, scale, axes=(0, None)), axes=(0,)), scale)


def _out(S: Any) -> Tuple:
    return Tuple(Array(S.N, S8), F32)


@composite("QuantRowS8", 1, ["N", "RELU"], lambda S: ((("x", Array(S.N, BF16)),), _out(S)), conformance=CONFORMANCE,
           refuse=refuse)
def QuantRowS8(B, S, x):
    """The row's s8 codes and f32 scale, relu'd first under RELU (the module docstring)."""
    f = _widened(B, S, x)
    return _codes(B, f, _scale(B, _amax(B, S, f)))


@composite("RowAbsMax", 1, ["N", "RELU"], lambda S: ((("x", Array(S.N, BF16)),), BF16), conformance=CONFORMANCE, refuse=refuse)
def RowAbsMax(B, S, x):
    """max |x_k| over the row, relu'd first under RELU, as a bf16 word (the module docstring)."""
    return B.call(SC.U32Hi16, _amax(B, S, _widened(B, S, x)))


@composite("QuantChunkS8", 1, ["N", "RELU"], lambda S: ((("x", Array(S.N, BF16)), ("amax", BF16)), _out(S)),
           conformance=CONFORMANCE, refuse=refuse)
def QuantChunkS8(B, S, x, amax):
    """A chunk's s8 codes under its row's amax, relu'd first under RELU, and the row's f32 scale (the module docstring)."""
    return _codes(B, _widened(B, S, x), _scale(B, B.call(P.Bf16ToF32, amax)))
