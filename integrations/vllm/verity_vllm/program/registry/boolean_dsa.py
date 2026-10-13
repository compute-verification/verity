"""`dsa`'s DeepSeek Sparse Attention indexer on bits: version 2 of each Definition, its word body line for line over Boolean
Definitions (the bodies are `dsa`'s, written once over `dsa.Ops`), so each one's word view is its `_v1` at the same statics.

    U32And_v2, I32Sub_v2, I32ToF32_v2, F32CeilFtz_v2     traced: and.b32, sub.s32, cvt.rn.f32.s32, cvt.rpi.ftz.f32.f32
    DsaUe8m0Scale_v2, DsaUe8m0Quant_v2{N}
    DsaIndexerKRow_v2{D,ROPE,EPS}, DsaIndexerQHead_v2{D,ROPE,NH}, DsaIndexerQ_v2{H,D,ROPE}
    DsaMqaLogit_v2{H,D}, DsaMqaLogits_v2{N,H,D}
    DsaTopKShortRow_v2{TOPK}                            traced at its TOPK: slot i's bits are i's, each one bit `len <= i`
    DsaTopKSelection_v2{TOPK,N}                         traced at its statics: a decoder per slot, unsigned keys

The callees are core's Boolean FP32 pieces (`verity.ml.circuits.ops.boolean`'s `elementwise`, `attention`, `cast`, `mufu`, `scalar`) and
`gemm.HopperE4m3QgmmaDot32_v2`; a constant word is constant bits and `-x` one NOT gate on x's sign bit (`F32Neg_v1`).  Each callee is
its word Definition on every word, NaN payloads included.  No table or lookup gate.

Each composite is one traced circuit (`flat`): its body over those callees, compiled (`verity.ml.circuits.evaluation.bits.compile`) and its
gates traced again (`trace.trace`), so a constant operand is folded and a value its calls share (the log2f polynomial's ``t``,
the scale every quantized element divides by, a row's mean) is computed once (`Q_word`'s no-recompute rule).
"""
from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from functools import reduce

from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.boolean import attention as BA
from verity.ml.circuits.ops.boolean import cast as BC
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.silicon import fp as FP
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import mufu as BM
from verity.ml.circuits.ops.boolean import scalar as SC
from verity.ml.circuits.boolean.forms import ONE, ZERO, NOT, OR, add, and_reduce, compress_columns, const_bits, mux, or_reduce, sub
from verity.ml.circuits.boolean.trace import definition
from verity_vllm.program.registry import dsa as D
from verity_vllm.program.registry.boolean_attention import _Bits
from verity_vllm.program.registry.topp_word_gates import I32ToF32 as I32ToF32V1

BIT = BL.BIT
BF16 = Array(16, BIT)
F32 = I32 = Array(32, BIT)
E4M3 = Array(8, BIT)
_AB = (("a", I32), ("b", I32))

U32AndV2 = definition("U32And", 2, lambda C, a, b: [C.AND(x, y) for x, y in zip(a, b)], _AB, I32, word=D.U32And,
                      doc="and.b32 on bits: U32And_v1")
I32SubV2 = definition("I32Sub", 2, lambda C, a, b: sub(C, a, b, 32)[0], _AB, I32, word=D.I32Sub,
                      doc="sub.s32 on bits: I32Sub_v1, a - b modulo 2^32")


def i32_to_f32(C, a):
    """`cvt.rn.f32.s32`: |a| (2^31 for INT_MIN) rounded to f32 at exponent 158 by `fp.round_f32`, the sign a's."""
    return FP.round_f32(C, a[31], const_bits(158, 9), 0, FP.neg_if(C, list(a), a[31]), ZERO, underflow=False)


I32ToF32V2 = definition("I32ToF32", 2, i32_to_f32, (("a", I32),), F32, word=I32ToF32V1,
                        doc="cvt.rn.f32.s32 on bits: I32ToF32_v1")


def f32_ceil_ftz(C, x):
    """`cvt.rpi.ftz.f32.f32` (`dsa.f32_ceil_ftz`): for 127 <= e < 150 the fraction mask is the 150 - e low bits (`fp.shl` of ones),
    the result x with them cleared, plus one past them when x is positive and any was set; below, +1.0 for a positive normal and
    the zero of the sign otherwise; from 150 x itself, a NaN 0x7FFFFFFF."""
    m, e, s = list(x[:23]), list(x[23:31]), x[31]
    nan = C.AND(and_reduce(C, e), or_reduce(C, m))
    lt127 = sub(C, e, const_bits(127, 8), 8)[1]
    lt150 = sub(C, e, const_bits(150, 8), 8)[1]
    keep = FP.shl(C, [ONE] * 23, sub(C, const_bits(150, 8), e, 8)[0][:5])
    frac = or_reduce(C, [C.AND(b, NOT(k)) for b, k in zip(m, keep)])
    trunc = [C.AND(b, k) for b, k in zip(m, keep)] + e
    up = add(C, [OR(C, b, NOT(k)) for b, k in zip(m, keep)] + e, [ONE], width=31)
    reg = mux(C, C.AND(NOT(s), frac), up, trunc)
    one = C.AND(NOT(s), or_reduce(C, e))
    small = [one if b.c else ZERO for b in const_bits(0x3F800000, 31)]
    out = mux(C, lt127, small, mux(C, lt150, reg, list(x[:31]))) + [s]
    return mux(C, nan, const_bits(0x7FFFFFFF, 32), out)


F32CeilFtzV2 = definition("F32CeilFtz", 2, f32_ceil_ftz, (("x", F32),), F32, word=D.F32CeilFtz,
                          doc="cvt.rpi.ftz.f32.f32 on bits: F32CeilFtz_v1")

BITS = dict(cvt=BC.Bf16ToF32, add=EW.F32Add, mul=EW.F32Mul, fma=EW.F32Fma, fma_ftz=BA.F32FmaFtz, div=EW.F32Div, rcp=BM.DivFullRcp,
            scale_a=BM.DivFullScaleA, rsqrt=BM.RsqrtApprox, ex2=BM.MufuEx2Ftz, fabs=SC.F32Fabs, fmaxf=SC.F32Fmaxf,
            e4m3=SC.F32ToE4m3Sat, iadd=EW.I32Add, iand=U32AndV2, isub=I32SubV2, i2f=I32ToF32V2, ceil=F32CeilFtzV2,
            isel=SC.SelectI32, dot=BG.HopperE4m3QgmmaDot32)


def bit_ops(B) -> D.Ops:
    c = _Bits(B)
    return D.Ops(B, BITS, lambda v: c.word(v, 32),
                 lambda x: Coll(F32, concat([x.refs.slice(0, 31), B.call(BL.NOT, x[31]).refs])))


def replay(c: BS.Compiled, C, args: list) -> list:
    """The compiled body ``c``'s gates on ``C`` over the forms ``args`` (its parameter leaves), its calls inlined: its return's
    forms."""
    W = [ZERO, ONE, *args] + [None] * (c.size - 2 - len(args))
    for step in c.steps:
        if step[0] == "gates":
            for kind, out, a, b in step[1]:
                for o, x, y in zip(out.tolist(), a.tolist(), b.tolist()):
                    W[o] = C.AND(W[x], W[y]) if kind == 0 else W[x] ^ W[y] if kind == 1 else NOT(W[x])
        elif step[0] == "call":
            _, child, members, base, w = step
            for m, row in enumerate(members.tolist()):
                W[base + m * w:base + (m + 1) * w] = replay(child, C, [W[i] for i in row])
        else:
            raise TypeError(f"{c.target}: a scan is not inlined")
    return [W[i] for i in c.ret_idx.tolist()]


def flat(name: str, version: int, statics: Sequence[str], signature: Callable, body: Callable, word,
         lifted: Sequence[str] = ()) -> CompositeDefinition:
    """The Boolean Definition ``name_v{version}{statics}`` whose circuit is ``body`` (over Boolean Definitions, as `EW.on_bits`
    takes it, its ``lifted`` statics Boolean Definitions read as their word views) compiled and traced as one circuit per binding
    (`EW.specialized`), its word view ``word`` at the same statics."""
    inner = EW.on_bits(name, version, statics, signature, body, word, register=False, lifted=lifted)

    def build(S):
        c = BS.compile(bind(inner, **{k: getattr(S, k) for k in statics}))
        widths = BL.elements(signature(S)[1])
        ends = [sum(widths[:i + 1]) for i in range(len(widths))]

        def run(C, *elements):
            out = replay(c, C, [f for e in elements for f in e])
            return [out[e - w:e] for w, e in zip(widths, ends)]
        return run

    return EW.specialized(name, version, statics, build, signature, doc=body.__doc__ or "", word=inner.word)


def _scale(B, S, amax):
    """`DsaUe8m0Scale_v1`'s body on bits."""
    return D.ue8m0_scale(bit_ops(B), amax)


DsaUe8m0ScaleV2 = flat("DsaUe8m0Scale", 2, [], lambda S: ((("amax", F32),), F32), _scale, D.DsaUe8m0Scale)


def _quant(B, S, x):
    """`DsaUe8m0Quant_v1`'s body on bits."""
    return D.ue8m0_quant(bit_ops(B), [x[i] for i in range(S.N)], DsaUe8m0ScaleV2, E4M3)


DsaUe8m0QuantV2 = flat("DsaUe8m0Quant", 2, ["N"], lambda S: ((("x", Array(S.N, F32)),), Tuple(Array(S.N, E4M3), F32)), _quant,
                             D.DsaUe8m0Quant)


def _k_row(B, S, k, w, b, cs):
    """`DsaIndexerKRow_v1`'s body on bits."""
    return D.k_slot(bit_ops(B), S, k, w, b, cs, bind(DsaUe8m0QuantV2, N=S.D), F32)


DsaIndexerKRowV2 = flat(
    "DsaIndexerKRow", 2, ["D", "ROPE", "EPS"],
    lambda S: ((("k", Array(S.D, BF16)), ("w", Array(S.D, F32)), ("b", Array(S.D, F32)), ("cs", Array(S.ROPE, BF16))),
               Tuple(Array(S.D, E4M3), F32)),
    _k_row, D.DsaIndexerKRow)


def _q_head(B, S, q, cs, w):
    """`DsaIndexerQHead_v1`'s body on bits."""
    return D.q_head(bit_ops(B), S, q, cs, w, bind(DsaUe8m0QuantV2, N=S.D), F32)


DsaIndexerQHeadV2 = flat(
    "DsaIndexerQHead", 2, ["D", "ROPE", "NH"],
    lambda S: ((("q", Array(S.D, BF16)), ("cs", Array(S.ROPE, BF16)), ("w", BF16)), Tuple(Array(S.D, E4M3), F32)), _q_head,
    D.DsaIndexerQHead)


def _q(B, S, q, cs, w):
    """`DsaIndexerQ_v1`'s body on bits."""
    return D.heads(B, bind(DsaIndexerQHeadV2, D=S.D, ROPE=S.ROPE, NH=S.H), S.H, S.D, q, cs, w, E4M3, F32)


DsaIndexerQV2 = flat(
    "DsaIndexerQ", 2, ["H", "D", "ROPE"],
    lambda S: ((("q", Array(S.H, Array(S.D, BF16))), ("cs", Array(S.ROPE, BF16)), ("w", Array(S.H, BF16))),
               Tuple(Array(S.H, Array(S.D, E4M3)), Array(S.H, F32))),
    _q, D.DsaIndexerQ)


def _logit(B, S, q, w, k, ks):
    """`DsaMqaLogit_v1`'s body on bits."""
    return D.mqa_logit(bit_ops(B), S.H, S.D, q, w, k, ks)


DsaMqaLogitV2 = flat(
    "DsaMqaLogit", 2, ["H", "D"],
    lambda S: ((("q", Array(S.H, Array(S.D, E4M3))), ("w", Array(S.H, F32)), ("k", Array(S.D, E4M3)), ("ks", F32)), F32),
    _logit, D.DsaMqaLogit)


def _logits(B, S, q, w, k, ks):
    """`DsaMqaLogits_v1`'s body on bits."""
    return B.batch(bind(DsaMqaLogitV2, H=S.H, D=S.D), q, w, k, ks, axes=(None, None, 0, 0))


DsaMqaLogitsV2 = flat(
    "DsaMqaLogits", 2, ["N", "H", "D"],
    lambda S: ((("q", Array(S.H, Array(S.D, E4M3))), ("w", Array(S.H, F32)), ("k", Array(S.N, Array(S.D, E4M3))),
                ("ks", Array(S.N, F32))), Array(S.N, F32)),
    _logits, D.DsaMqaLogits)


def _topk(S):
    """`DsaTopKShortRow_v1{TOPK}` traced: `len <= i` (signed: the borrow of `(i ^ 2^31) - (len ^ 2^31)` negated) is one bit per
    slot, and slot i is i's bits each ORed with it (-1 when set)."""
    def build(C, n):
        flip = list(n[:31]) + [NOT(n[31])]
        out = []
        for i in range(S.TOPK):
            ci = const_bits(i ^ (1 << 31), 32)
            le = NOT(sub(C, ci, flip, 32)[1])
            out.append([ONE if b else le for b in ((i >> j) & 1 for j in range(32))])
        return out
    return build


DsaTopKShortRowV2 = EW.specialized("DsaTopKShortRow", 2, ["TOPK"], _topk, lambda S: ((("len", I32),), Array(S.TOPK, I32)),
                                   doc=_topk.__doc__, word=lambda b: bind(D.DsaTopKShortRow, TOPK=b["TOPK"]))


def decode(C, bits: list, n: int) -> list:
    """The first n lines of the decoder of the unsigned index ``bits`` (n <= 2^len(bits)): line j is `bits == j`, one AND of a line
    of the low half's decoder and one of the high half's."""
    if len(bits) == 1:
        return [NOT(bits[0]), bits[0]][:n]
    h = len(bits) // 2
    lo, hi = decode(C, bits[:h], min(n, 1 << h)), decode(C, bits[h:], -(-n >> h))
    return [C.AND(hi[j >> h], lo[j & ((1 << h) - 1)]) for j in range(n)]


def _selection(S):
    """`DsaTopKSelection_v1{TOPK, N}` traced.  `len <= j` is `_topk`'s comparison; keys compare unsigned as `order_key ^ 2^31`
    (`convert_to_uint32_v2`: bit 31 the sign's NOT, bit b < 31 x_b ^ x_31), the sentinels INT_MAX and INT_MIN all ones and all
    zeros; `idx < len` unsigned is the range check (len > TOPK > 0 wherever it is read).  Slot i's membership lines decode its low
    ceil(log2 N) bits (an index in range has no other) and position j's membership is their XOR over the slots: with every index
    in range, TOPK odd positions means TOPK distinct indices, as TOPK members by the OR does, so the bit is v1's.  The count is
    `compress_columns`."""
    D.check_selection_statics(S.TOPK, S.N)
    T, N = S.TOPK, S.N
    p = (N - 1).bit_length()

    def le_u(C, a, b):
        return NOT(sub(C, b, a, 32)[1])

    def build(C, *els):
        x, n, idx = els[:N], els[N], els[N + 1:]
        flip = list(n[:31]) + [NOT(n[31])]
        le = [NOT(sub(C, const_bits(j ^ (1 << 31), 32), flip, 32)[1]) for j in range(N + 1)]
        short = and_reduce(C, [b if (i >> k) & 1 else NOT(b ^ le[i]) for i in range(T) for k, b in enumerate(idx[i])])
        inside = and_reduce(C, [sub(C, idx[i], n, 32)[1] for i in range(T)])
        lines = [decode(C, list(idx[i][:p]), N) for i in range(T)]
        member = [reduce(operator.xor, (lines[i][j] for i in range(T))) for j in range(N)]
        count = compress_columns(C, [member])
        tight = and_reduce(C, [c if (T >> k) & 1 else NOT(c) for k, c in enumerate(count)])
        key = [[b ^ v[31] for b in v[:31]] + [NOT(v[31])] for v in x]
        least = D.pairwise([mux(C, member[j], key[j], const_bits(D.M32, 32)) for j in range(N)],
                           lambda a, b: mux(C, le_u(C, a, b), a, b))
        most = D.pairwise([mux(C, OR(C, member[j], le[j]), const_bits(0, 32), key[j]) for j in range(N)],
                          lambda a, b: mux(C, le_u(C, a, b), b, a))
        long_ok = and_reduce(C, [inside, tight, le_u(C, most, least), le[N]])
        return [OR(C, C.AND(le[T], short), C.AND(NOT(le[T]), long_ok))]
    return build


DsaTopKSelectionV2 = EW.specialized(
    "DsaTopKSelection", 2, ["TOPK", "N"], _selection,
    lambda S: ((("logits", Array(S.N, F32)), ("len", I32), ("idx", Array(S.TOPK, I32))), BIT),
    doc=_selection.__doc__, word=lambda b: bind(D.DsaTopKSelection, TOPK=b["TOPK"], N=b["N"]))

BOOLEAN_DSA = [U32AndV2, I32SubV2, I32ToF32V2, F32CeilFtzV2, DsaUe8m0ScaleV2, DsaUe8m0QuantV2, DsaIndexerKRowV2, DsaIndexerQHeadV2,
               DsaIndexerQV2, DsaMqaLogitV2, DsaMqaLogitsV2, DsaTopKShortRowV2, DsaTopKSelectionV2]
