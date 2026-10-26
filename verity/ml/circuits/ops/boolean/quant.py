"""The per-row s8 quantizer on bits: the next version of each word Definition of `ops.quant`, the same function.

    QuantRowS8_v2{N, RELU}      (x: Array<N, Array<16, Value<1>>>) -> (Array<N, Array<8, Value<1>>>, Array<32, Value<1>>)
                                                                                          word view: QuantRowS8_v1{N, RELU}
    RowAbsMax_v2{N, RELU}       (x: Array<N, Array<16, Value<1>>>) -> Array<16, Value<1>>   word view: RowAbsMax_v1{N, RELU}
    QuantChunkS8_v2{N, RELU}    (x: Array<N, Array<16, Value<1>>>, amax: Array<16, Value<1>>)
                                -> (Array<N, Array<8, Value<1>>>, Array<32, Value<1>>)    word view: QuantChunkS8_v1{N, RELU}

Each is one circuit traced from builders at its statics (`elementwise.specialized`), so its constants (bf16 +0, 127.0, 1.0) are
folded into its gates and the gates its outputs don't read are dropped. The ReLU is `scalar.bf16_gt_strict` against +0 and
`scalar.select`. The absmax is the unsigned max (`scalar.gt_u`) of the relu'd words' 15 magnitude bits: the widening to f32 is
exact and keeps the order, NaNs above +inf, so it is the word Definition's max of the widened magnitudes, and its high half the
bf16 word. The scale's amax > 0 is the sign bit clear and the magnitude neither zero nor a NaN; the divisions are `fp.f32_div`
with the IR's NaN words (`F32Div_v3`'s, `F32Div_v1` on bits) and the codes `scalar.f32_to_s8_rni_sat`.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ZERO, and_reduce, const_bits, mux, or_reduce
from verity.ml.circuits.defs import bind
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array, Tuple

from .. import quant as Q
from .elementwise import specialized
from .scalar import bf16_gt_strict, f32_to_s8_rni_sat, gt_u, select

__all__ = ["QuantChunkS8", "QuantRowS8", "RowAbsMax"]

BF16 = Array(16, BL.BIT)
CODE = Array(8, BL.BIT)
F32 = Array(32, BL.BIT)
_ZERO16 = const_bits(0, 16)


def _relu(C, S, x):
    x = [list(g) for g in x]
    return [select(C, bf16_gt_strict(C, _ZERO16, g)[0], _ZERO16, g) for g in x] if S.RELU else x


def _amax(C, x):
    """The bf16 amax of relu'd words."""
    amax = x[0][:15]
    for g in x[1:]:
        amax = mux(C, gt_u(C, g[:15], amax), g[:15], amax)
    return amax + [ZERO]


def _codes(C, x, amax):
    """The codes of relu'd words under amax's scale, then the scale: RN(amax / 127) when amax > 0, else 1.0."""
    amax = list(amax)
    nan = C.AND(and_reduce(C, amax[7:15]), or_reduce(C, amax[:7]))
    pos = C.AND(NOT(amax[15]), C.AND(or_reduce(C, amax[:15]), NOT(nan)))
    div = fp.f32_div(C, fp.bf16_to_f32(amax), fp.f32_const(Q.F32_127), ir_nan=True)
    scale = mux(C, pos, div, fp.f32_const(Q.F32_ONE))
    return [f32_to_s8_rni_sat(C, fp.f32_div(C, fp.bf16_to_f32(g), scale, ir_nan=True)) for g in x] + [scale]


def _row(S):
    def build(C, *x):
        x = _relu(C, S, x)
        return _codes(C, x, _amax(C, x))
    return build


def _absmax(S):
    def build(C, *x):
        return _amax(C, _relu(C, S, x))
    return build


def _chunk(S):
    def build(C, *e):
        return _codes(C, _relu(C, S, e[:-1]), e[-1])
    return build


def _out(S) -> Tuple:
    return Tuple(Array(S.N, CODE), F32)


QuantRowS8 = specialized("QuantRowS8", 2, ("N", "RELU"), _row, lambda S: ((("x", Array(S.N, BF16)),), _out(S)),
                         doc="`QuantRowS8_v1`'s codes and scale of the row on bits, one circuit from its builders",
                         word=lambda b: bind(Q.QuantRowS8, **b))
RowAbsMax = specialized("RowAbsMax", 2, ("N", "RELU"), _absmax, lambda S: ((("x", Array(S.N, BF16)),), BF16),
                        doc="`RowAbsMax_v1`'s row absmax on bits, one circuit from its builders",
                        word=lambda b: bind(Q.RowAbsMax, **b))
QuantChunkS8 = specialized("QuantChunkS8", 2, ("N", "RELU"), _chunk,
                           lambda S: ((("x", Array(S.N, BF16)), ("amax", BF16)), _out(S)),
                           doc="`QuantChunkS8_v1`'s codes of the chunk under the row's amax, and the scale, on bits, one "
                               "circuit from its builders",
                           word=lambda b: bind(Q.QuantChunkS8, **b))
QuantRowS8.refuse = RowAbsMax.refuse = QuantChunkS8.refuse = Q.refuse  # type: ignore[attr-defined]
