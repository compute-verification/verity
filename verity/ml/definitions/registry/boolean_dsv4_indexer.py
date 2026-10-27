"""`dsv4_indexer`'s Definitions on bits: version 2 of each, its word body line for line over gates (the bodies are
`dsv4_indexer`'s, written once over its steps), so each one's word view is its `_v1` at the same statics.

    Dsv4CompressorScore_v2{W}, Dsv4CompressColumn_v2{ROWS}, Dsv4IndexerNormRope_v2{D,ROPE,EPS}, Dsv4IndexerKQuant_v2{N},
    Dsv4InvRope_v2{ROPE}, Dsv4InvRopeQuant_v2{N}       traced at their statics (`EW.specialized`)
    Dsv4IndexerCompress_v2{D,ROPE,EPS}                 Calls of the column, norm and quantisation (`EW.on_bits`)
    Dsv4InvRopeQuantHead_v2{CHUNKS,QG,ROPE}            its Calls of `Bf16ToF32_v2`, the inverse RoPE and the group quantisation
                                                       traced as one circuit (`boolean_dsa.flat`): a widened word's 16 zero
                                                       bits fold instead of reaching a separately traced callee
    Dsv4ShortContextTopK_v2{TOPK,RATIO}                traced: one validity bit per slot from pos's bits

A step is core's gate builder for the word primitive `dsv4_indexer.WORD` names: `fp.f32_add` / `f32_sub` / `f32_mul` / `f32_fma`
with the GPU's NaN word (the `_v2` FP32 primitives, as C-Flock lowers them), the FTZ FMA flushed around `fp.f32_fma`, `mufu`'s
MUFU pieces, `scalar`'s max, min and e4m3 conversion, `boolean_dsa`'s `cvt.rn.f32.s32` and `cvt.rpi.ftz`, `forms`' integer add,
subtract and compare; a constant word is constant bits, so it folds into the gates.  No table or lookup gate.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array
from verity.ml.circuits.ops.boolean import cast as BC
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import mufu as BM
from verity.ml.circuits.ops.boolean import scalar as SC
from verity.ml.circuits.ops.boolean.softmax import f32_gt, ftz
from verity.ml.circuits.silicon import fp as FP
from verity.ml.circuits.boolean.forms import ONE, ZERO, NOT, OR, add, and_reduce, const_bits, mux, or_reduce, sub
from verity.ml.definitions.registry import boolean_dsa as BD
from verity.ml.definitions.registry import dsv4_indexer as I

BIT = BL.BIT
BITS = SimpleNamespace(F32=Array(32, BIT), BF16=Array(16, BIT), E4M3=Array(8, BIT), I32=Array(32, BIT), I64=Array(64, BIT))
PTX = FP.F32_NAN_PTX


class GateOps:
    """`dsv4_indexer`'s steps as gates on ``C``: each value a list of forms (LSB first), each method the gates of the word primitive
    of the same name in `dsv4_indexer.WORD`."""

    def __init__(self, C):
        self.C = C

    def word(self, bits: int):
        return const_bits(bits, 32)

    def neg(self, x):
        return FP.f32_neg(x)

    def add(self, a, b):
        return FP.f32_add(self.C, a, b, nan_word=PTX)

    def sub(self, a, b):
        return FP.f32_sub(self.C, a, b, nan_word=PTX)

    def mul(self, a, b):
        return FP.f32_mul(self.C, a, b, nan_word=PTX)

    def fma(self, a, b, c):
        return FP.f32_fma(self.C, a, b, c, nan_word=PTX)

    def fma_ftz(self, a, b, c):
        C = self.C
        return ftz(C, FP.f32_fma(C, ftz(C, a), ftz(C, b), ftz(C, c), nan_word=PTX))

    def rcp(self, b):
        return BM.div_full_rcp(self.C, b)

    def scale_a(self, a, b):
        return BM.div_full_scale_a(self.C, a, b)

    def rsqrt(self, x):
        return BM.rsqrt_approx(self.C, x)

    def ex2(self, x):
        return BM.mufu_ex2(self.C, x)

    def fabs(self, x):
        return list(x[:31]) + [ZERO]

    def fmaxf(self, a, b):
        return SC.fmaxmin(self.C, a, b, True)

    def fminf(self, a, b):
        return SC.fmaxmin(self.C, a, b, False)

    def e4m3(self, x):
        return SC.f32_to_e4m3_sat(self.C, x)

    def iadd(self, a, b):
        return add(self.C, a, b, width=32)

    def iand(self, a, b):
        return [self.C.AND(x, y) for x, y in zip(a, b)]

    def isub(self, a, b):
        return sub(self.C, a, b, 32)[0]

    def i2f(self, a):
        return BD.i32_to_f32(self.C, a)

    def ceil(self, x):
        return BD.f32_ceil_ftz(self.C, x)

    def cvt(self, h):
        return FP.bf16_to_f32(h)

    def bf16(self, u):
        return FP.f32_to_bf16(self.C, u, 0x7FFF)

    def fsel(self, c, a, b):
        return mux(self.C, c, a, b)

    def gt(self, a, b):
        return f32_gt(self.C, a, b)[0]

    def ile(self, a, b):
        """Signed a <= b: not b < a, the borrow of (b ^ 2^31) - (a ^ 2^31)."""
        flip = lambda x: list(x[:31]) + [NOT(x[31])]  # noqa: E731
        return NOT(sub(self.C, flip(b), flip(a), 32)[1])


def traced(name: str, statics: list[str], signature, body, word):
    """The Boolean Definition ``name_v2{statics}``: `dsv4_indexer`'s ``body`` over `GateOps`, traced per binding, its word view
    ``word`` at the same statics."""
    sig = signature(BITS)

    def build(S):
        counts = [len(BL.elements(t)) for _, t in sig(S)[0]]

        def run(C, *elements):
            parts, i = [], 0
            for n in counts:
                parts.append(list(elements[i:i + n]))
                i += n
            return body(GateOps(C), S, *parts)
        return run

    return EW.specialized(name, 2, statics, build, sig, doc=f"`{word.name}_v1`'s body on bits: {(body.__doc__ or '').strip()}",
                          word=lambda b: bind(word, **b))


Dsv4CompressorScoreV2 = traced("Dsv4CompressorScore", ["W"], I.score_sig, I.compressor_score, I.Dsv4CompressorScore)
Dsv4CompressColumnV2 = traced("Dsv4CompressColumn", ["ROWS"], I.column_sig, I.compress_column, I.Dsv4CompressColumn)
Dsv4IndexerNormRopeV2 = traced("Dsv4IndexerNormRope", ["D", "ROPE", "EPS"], I.norm_rope_sig, I.norm_rope, I.Dsv4IndexerNormRope)
Dsv4IndexerKQuantV2 = traced("Dsv4IndexerKQuant", ["N"], I.quant_sig, I.k_quant, I.Dsv4IndexerKQuant)
Dsv4InvRopeV2 = traced("Dsv4InvRope", ["ROPE"], I.inv_rope_sig, I.inv_rope, I.Dsv4InvRope)
Dsv4InvRopeQuantV2 = traced("Dsv4InvRopeQuant", ["N"], I.quant_sig, I.inv_rope_quant, I.Dsv4InvRopeQuant)

BIT_DEFS = SimpleNamespace(column=Dsv4CompressColumnV2, norm_rope=Dsv4IndexerNormRopeV2, k_quant=Dsv4IndexerKQuantV2,
                           cvt=BC.Bf16ToF32, inv_rope=Dsv4InvRopeV2, quant=Dsv4InvRopeQuantV2)


def _compress(B, S, kv, score, w, cs):
    """`Dsv4IndexerCompress_v1`'s body on bits: Calls of `Dsv4CompressColumn_v2{8}` per column, `Dsv4IndexerNormRope_v2` and
    `Dsv4IndexerKQuant_v2{D}`."""
    return I.compress(B, S, kv, score, w, cs, BIT_DEFS, BITS)


Dsv4IndexerCompressV2 = EW.on_bits("Dsv4IndexerCompress", 2, ["D", "ROPE", "EPS"], I.compress_sig(BITS), _compress,
                                   I.Dsv4IndexerCompress)


def _head(B, S, x, cs):
    """`Dsv4InvRopeQuantHead_v1`'s body on bits: `Bf16ToF32_v2`, `Dsv4InvRope_v2{ROPE}` and `Dsv4InvRopeQuant_v2{QG}`, traced as
    one circuit."""
    return I.inv_rope_head(B, S, x, cs, BIT_DEFS, BITS)


Dsv4InvRopeQuantHeadV2 = BD.flat("Dsv4InvRopeQuantHead", 2, ["CHUNKS", "QG", "ROPE"], I.head_sig(BITS), _head,
                                 I.Dsv4InvRopeQuantHead)


def _topk(S):
    """`Dsv4ShortContextTopK_v1{TOPK, RATIO}` traced: ok = not b63 and not all of b0..b62, big = any of b_L..b62, and slot i's
    validity `ok and (big or not (low L bits < R(i+1) - 1))`; slot i is i's bits each ORed with its invalidity (-1 when set)."""
    L = I.check_topk_statics(S.TOPK, S.RATIO)

    def build(C, pos):
        ok = C.AND(NOT(pos[63]), NOT(and_reduce(C, list(pos[:63]))))
        big = or_reduce(C, list(pos[L:63]))
        out = []
        for i in range(S.TOPK):
            lt = sub(C, list(pos[:L]), const_bits(S.RATIO * (i + 1) - 1, L), L)[1] if L else ZERO
            bad = NOT(C.AND(ok, OR(C, big, NOT(lt))))
            out.append([ONE if (i >> j) & 1 else bad for j in range(32)])
        return out
    return build


Dsv4ShortContextTopKV2 = EW.specialized("Dsv4ShortContextTopK", 2, ["TOPK", "RATIO"], _topk, I.topk_sig(BITS), doc=_topk.__doc__,
                                        word=lambda b: bind(I.Dsv4ShortContextTopK, **b))

BOOLEAN_DSV4_INDEXER = [Dsv4CompressorScoreV2, Dsv4CompressColumnV2, Dsv4IndexerNormRopeV2, Dsv4IndexerKQuantV2,
                        Dsv4IndexerCompressV2, Dsv4InvRopeV2, Dsv4InvRopeQuantV2, Dsv4InvRopeQuantHeadV2, Dsv4ShortContextTopKV2]
