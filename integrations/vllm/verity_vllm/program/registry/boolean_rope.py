"""vLLM's NeoX rotation (`_C.rotary_embedding`) as Boolean Definitions: the next version of each word Definition, the same
function on bits.

    RopeOut_v2, RopeOutAdd_v2   (x, y, c, s: Array<16, Value<1>>) -> Array<16, Value<1>>   word view: RopeOut_v1, RopeOutAdd_v1
    RoPEHead_v2{D}              (x, cs: Array<D, Array<16, Value<1>>>) -> Array<D, Array<16, Value<1>>>
                                                                                          word view: RoPEHead_v1{D}
    RoPE_v2{NHEADS, D}          (x: Array<NHEADS, Array<D, Array<16, Value<1>>>>, cs: Array<D, Array<16, Value<1>>>)
                                -> Array<NHEADS, Array<D, Array<16, Value<1>>>>           word view: RoPE_v1{NHEADS, D}

The rotations are traced from core's f32 builders (`verity.ml.boolean.fp`, `trace.definition`), the pieces C-Flock's lowering
runs for the word primitives: ``t = RN32(y * s)`` (FMUL, subnormals kept), ``RN32(x * c - t)`` with one FFMA rounding, then
bf16_rn with aten's NaN 0x7FC0. Every NaN of the f32 steps ends in that cast, so their payloads never reach the output. The
composites are `b1`'s bodies on bits: `RoPEHead_v2` batches `RopeOut_v2` and `RopeOutAdd_v2` over the D/2 coordinate pairs and
`RoPE_v2` batches `RoPEHead_v2` over the heads with one shared cos/sin row, so both are separable and each output element is
one proof unit under `Q_word` v1, as in the word Program.
"""
from __future__ import annotations

from verity.ir import boolean as BL
from verity.ir.defs import CompositeDefinition, bind
from verity.ir.refs import Coll, concat
from verity.ir.types import Array
from verity.ml.boolean import fp
from verity.ml.boolean.trace import definition

from . import b1
from . import prims as P

__all__ = ["RoPE", "RoPEHead", "RopeOut", "RopeOutAdd"]

BF16 = Array(16, BL.BIT)
_XYCS = (("x", BF16), ("y", BF16), ("c", BF16), ("s", BF16))


def rope_out(C, x, y, c, s):
    """``bf16_rn(RN32(x * c - RN32(y * s)))``, NaN 0x7FC0 (`prims.RopeOut`)."""
    t = fp.f32_mul(C, fp.bf16_to_f32(y), fp.bf16_to_f32(s))
    return fp.f32_to_bf16(C, fp.f32_fma(C, fp.bf16_to_f32(x), fp.bf16_to_f32(c), fp.f32_neg(t)), P.BF16_NAN_TORCH)


def rope_out_add(C, x, y, c, s):
    """``bf16_rn(RN32(y * c + RN32(x * s)))``, NaN 0x7FC0 (`prims.RopeOutAdd`)."""
    t = fp.f32_mul(C, fp.bf16_to_f32(x), fp.bf16_to_f32(s))
    return fp.f32_to_bf16(C, fp.f32_fma(C, fp.bf16_to_f32(y), fp.bf16_to_f32(c), t), P.BF16_NAN_TORCH)


RopeOut = definition("RopeOut", 2, rope_out, _XYCS, BF16, word=P.RopeOut,
                     doc="NeoX RoPE's first half on bits: RopeOut_v1, bf16_rn(RN32(x*c - RN32(y*s)))")
RopeOutAdd = definition("RopeOutAdd", 2, rope_out_add, _XYCS, BF16, word=P.RopeOutAdd,
                        doc="NeoX RoPE's second half on bits: RopeOutAdd_v1, bf16_rn(RN32(y*c + RN32(x*s)))")


def _rope_head(B, S, x, cs):
    """`RoPEHead_v1`'s body on bits, ``cs = [cos[0:R] | sin[0:R]]`` and R = D/2: ``out[i] = RopeOut_v2(x[i], x[i+R], cos[i],
    sin[i])``, ``out[i+R] = RopeOutAdd_v2(...)`` on the same operands."""
    R = S.D // 2
    cos, sin = cs[0:R], cs[R:S.D]
    lo = B.batch(RopeOut, x[0:R], x[R:S.D], cos, sin, axes=(0, 0, 0, 0))
    hi = B.batch(RopeOutAdd, x[0:R], x[R:S.D], cos, sin, axes=(0, 0, 0, 0))
    return Coll(Array(S.D, BF16), concat([lo.refs, hi.refs]))


def _rope(B, S, x, cs):
    """`RoPE_v1`'s body on bits: `RoPEHead_v2{D}` on each head, the cos/sin row shared."""
    return B.batch(bind(RoPEHead, D=S.D), x, cs, axes=(0, None))


# Constructed rather than decorated so that setting `word` reads as construction to the P9 lint, not as a runtime patch.
RoPEHead = CompositeDefinition("RoPEHead", 2, ("D",),
                               lambda S: ((("x", Array(S.D, BF16)), ("cs", Array(S.D, BF16))), Array(S.D, BF16)),
                               _rope_head, doc=_rope_head.__doc__,
                               conformance="exact-model-tested; qualification: " + b1.COS_SIN_QUALIFICATION)
RoPEHead.word = lambda b: bind(b1.RoPEHead, D=b["D"])  # type: ignore[attr-defined]
RoPE = CompositeDefinition("RoPE", 2, ("NHEADS", "D"),
                           lambda S: ((("x", Array(S.NHEADS, Array(S.D, BF16))), ("cs", Array(S.D, BF16))),
                                      Array(S.NHEADS, Array(S.D, BF16))),
                           _rope, doc=_rope.__doc__,
                           conformance="exact-model-tested; qualification: " + b1.COS_SIN_QUALIFICATION)
RoPE.word = lambda b: bind(b1.RoPE, NHEADS=b["NHEADS"], D=b["D"])  # type: ignore[attr-defined]
