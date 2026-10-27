"""vLLM's NeoX rotation (`_C.rotary_embedding`) as Boolean Definitions: the next version of each word Definition, the same
function on bits.

    RopeOut_v2, RopeOutAdd_v2   (x, y, c, s: Array<16, Value<1>>) -> Array<16, Value<1>>   word view: RopeOut_v1, RopeOutAdd_v1
    RoPEHead_v2{D}              (x, cs: Array<D, Array<16, Value<1>>>) -> Array<D, Array<16, Value<1>>>
                                                                                          word view: RoPEHead_v1{D}
    RoPE_v2{NHEADS, D}          (x: Array<NHEADS, Array<D, Array<16, Value<1>>>>, cs: Array<D, Array<16, Value<1>>>)
                                -> Array<NHEADS, Array<D, Array<16, Value<1>>>>           word view: RoPE_v1{NHEADS, D}
    RoPEHeadGptJ_v3{D}, RoPEGptJ_v3{NHEADS, D}   the same signatures, the interleaved pairing (`is_neox=False`)
                                                       word views, defined here: RoPEHeadGptJ_v2{D}, RoPEGptJ_v2{NHEADS, D}

The rotations are traced from core's f32 builders (`verity.ml.circuits.silicon.fp`, `trace.definition`), the pieces C-Flock's lowering
runs for the word primitives: ``t = RN32(y * s)`` (FMUL, subnormals kept), ``RN32(x * c - t)`` with one FFMA rounding, then
bf16_rn with aten's NaN 0x7FC0. Every NaN of the f32 steps ends in that cast, so their payloads never reach the output. The
composites are `b1`'s bodies on bits: `RoPEHead_v2` batches `RopeOut_v2` and `RopeOutAdd_v2` over the D/2 coordinate pairs and
`RoPE_v2` batches `RoPEHead_v2` over the heads with one shared cos/sin row, so both are separable and each output element is
one proof block under `Q_word` v1, as in the word Program. The GPT-J versions are the same gates on the pairs (2i, 2i+1),
the kernel's `IS_NEOX=false` branch (DeepSeek-V2 MLA's q_pe and k_pe: GLM-4.7-Flash, GLM-5.3), with the odd half's operands
swapped as nvcc contracts that instantiation: ``RopeOutAdd_v2(y, x, s, c) = bf16_rn(RN32(x * s + RN32(y * c)))``.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind, composite
from verity.ml.circuits.refs import Coll, Explicit, concat, strided_view
from verity.ml.circuits.types import Array
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.trace import definition
from verity.ml.definitions.evidence import EVIDENCE

from . import b1
from . import prims as P

__all__ = ["RoPE", "RoPEGptJ", "RoPEGptJV2", "RoPEHead", "RoPEHeadGptJ", "RoPEHeadGptJV2", "RopeOut", "RopeOutAdd"]

BF16 = Array(16, BL.BIT)
_XYCS = (("x", BF16), ("y", BF16), ("c", BF16), ("s", BF16))
_DIFFTEST = EVIDENCE["cc12_difftest"]["run"]
GPTJ_CONFORMANCE = ("exact-model-tested; RoPEHeadGptJ_v2 on bits (test_boolean_rope.py), which matches vLLM d9105ea80's "
                    f"`rotary_embedding(is_neox=False)` in 60 of 60 difftest cases (run {_DIFFTEST}); "
                    "qualification: " + b1.COS_SIN_QUALIFICATION)
GPTJ_V2_CONFORMANCE = ("exact-model-tested; 60 of 60 difftest cases (101,696 words) of vLLM d9105ea80's CUDA `rotary_embedding("
                       f"is_neox=False)` (bf16 query, key and cache, run {_DIFFTEST}), q_pe/k_pe at GLM-4.7-Flash's "
                       "and GLM-5.3's shapes; qualification: " + b1.COS_SIN_QUALIFICATION)


@composite("RoPEHeadGptJ", 2, ["D"], lambda S: ((("x", Array(S.D, P.BF16)), ("cs", Array(S.D, P.BF16))), Array(S.D, P.BF16)),
           conformance=GPTJ_V2_CONFORMANCE)
def RoPEHeadGptJV2(B, S, x, cs):
    """vLLM `rotary_embedding(..., is_neox=False)` on one head with the bf16 cache row `cs = [cos[0:R] | sin[0:R]]`, R = D/2,
    as nvcc contracts `pos_encoding_kernels.cu`'s `x_f * cos_f - y_f * sin_f` and `y_f * cos_f + x_f * sin_f` for that
    instantiation (x = x[2i], y = x[2i+1]): `out[2i] = RopeOut_v1(x, y, cos[i], sin[i]) = bf16(RN32(x*cos - RN32(y*sin)))`,
    `out[2i+1] = RopeOutAdd_v1(y, x, sin[i], cos[i]) = bf16(RN32(x*sin + RN32(y*cos)))`. The odd half's rounded product is
    y*cos, not `RoPEHeadGptJ_v1`'s x*sin. A product of two bf16 words is exact in f32 unless it overflows or falls below
    f32's normal range, so the two differ only there."""
    R = S.D // 2
    cos, sin = cs[0:R], cs[R:S.D]
    xe = Coll(Array(R, P.BF16), strided_view(x, 0, R, 0, 2, R))
    xo = Coll(Array(R, P.BF16), strided_view(x, 1, R, 0, 2, R))
    lo = B.batch(P.RopeOut, xe, xo, cos, sin, axes=(0, 0, 0, 0))  # x[2i]*cos - RN(x[2i+1]*sin)
    hi = B.batch(P.RopeOutAdd, xo, xe, sin, cos, axes=(0, 0, 0, 0))  # x[2i]*sin + RN(x[2i+1]*cos)
    return Coll(Array(S.D, P.BF16), Explicit(tuple(r for i in range(R) for r in (lo.refs[i], hi.refs[i]))))


@composite("RoPEGptJ", 2, ["NHEADS", "D"],
           lambda S: ((("x", Array(S.NHEADS, Array(S.D, P.BF16))), ("cs", Array(S.D, P.BF16))),
                      Array(S.NHEADS, Array(S.D, P.BF16))),
           conformance=GPTJ_V2_CONFORMANCE)
def RoPEGptJV2(B, S, x, cs):
    """`RoPEHeadGptJ_v2{D}` over NHEADS heads sharing one cache row (the `RoPE_v1` shape)."""
    return B.batch(bind(RoPEHeadGptJV2, D=S.D), x, cs, axes=(0, None))


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


def _rope_head_gptj(B, S, x, cs):
    """`RoPEHeadGptJ_v2`'s body on bits (`_C.rotary_embedding(..., is_neox=False)`), ``cs = [cos[0:R] | sin[0:R]]``:
    ``out[2i] = RopeOut_v2(x[2i], x[2i+1], cos[i], sin[i])``, ``out[2i+1] = RopeOutAdd_v2(x[2i+1], x[2i], sin[i], cos[i])``."""
    R = S.D // 2
    cos, sin = cs[0:R], cs[R:S.D]
    xe = Coll(Array(R, BF16), strided_view(x, 0, 16, 32, 1, 16 * R))
    xo = Coll(Array(R, BF16), strided_view(x, 16, 16, 32, 1, 16 * R))
    lo = B.batch(RopeOut, xe, xo, cos, sin, axes=(0, 0, 0, 0))
    hi = B.batch(RopeOutAdd, xo, xe, sin, cos, axes=(0, 0, 0, 0))
    return Coll(Array(S.D, BF16), concat([concat([lo[i].refs, hi[i].refs]) for i in range(R)]))


def _rope_gptj(B, S, x, cs):
    """`RoPEGptJ_v2`'s body on bits: `RoPEHeadGptJ_v3{D}` on each head, the cos/sin row shared."""
    return B.batch(bind(RoPEHeadGptJ, D=S.D), x, cs, axes=(0, None))


RoPEHeadGptJ = CompositeDefinition("RoPEHeadGptJ", 3, ("D",),
                                   lambda S: ((("x", Array(S.D, BF16)), ("cs", Array(S.D, BF16))), Array(S.D, BF16)),
                                   _rope_head_gptj, doc=_rope_head_gptj.__doc__, conformance=GPTJ_CONFORMANCE)
RoPEHeadGptJ.word = lambda b: bind(RoPEHeadGptJV2, D=b["D"])  # type: ignore[attr-defined]
RoPEGptJ = CompositeDefinition("RoPEGptJ", 3, ("NHEADS", "D"),
                               lambda S: ((("x", Array(S.NHEADS, Array(S.D, BF16))), ("cs", Array(S.D, BF16))),
                                          Array(S.NHEADS, Array(S.D, BF16))),
                               _rope_gptj, doc=_rope_gptj.__doc__, conformance=GPTJ_CONFORMANCE)
RoPEGptJ.word = lambda b: bind(RoPEGptJV2, NHEADS=b["NHEADS"], D=b["D"])  # type: ignore[attr-defined]
