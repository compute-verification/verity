"""`dsv4_qnorm_insert`'s DeepSeek-V4-Pro query norm, RoPE and SWA insert, and the indexer's query RoPE and FP8 quantisation, on
bits: version 2 of each Definition, its word body line for line over Boolean Definitions (the bodies are `dsv4_qnorm_insert`'s,
written once over its `Ops`), so each one's word view is its `_v1` at the same statics.

    F32ToU32RzByte_v2                                 traced: the low byte of cvt.rzi.u32.f32
    Dsv4QNormRope_v2{D,DR,EPS}
    Dsv4KvRopeQuant_v2{DN,DR,QB}
    Dsv4QNormRopeKvInsert_v2{NH,NHP,DN,DR,QB,EPS}
    Dsv4IndexerQHead_v2{D,DR,SCALE}
    Dsv4IndexerQ_v2{H,D,DR,SCALE}

The callees are core's Boolean pieces: `Bf16ToF32_v2`, `F2fpBf16_v2`, the GPU-NaN f32 operations `F32Add_v4` / `F32Mul_v4` /
`F32Fma_v4` / `F32Div_v4`, `RsqrtApprox_v2`, `F32Fabs_v2`, `F32Fmaxf_v2`, `F32Fminf_v2`, `F32ToE4m3Sat_v2`, `I32Add_v2`,
`boolean_dsa`'s `U32And_v2`, `I32Sub_v2` and `F32CeilFtz_v2`, and `boolean_flashmla`'s `NvLog2f_v2` and `NvExp2f_v2`; a constant
word is constant bits and `-x` one NOT gate on x's sign bit.  No table or lookup gate.  Each composite is one traced circuit
(`boolean_dsa.flat`).
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ONE, ZERO, const_bits, or_reduce, sub
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.ops.boolean import cast as BC
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import mufu as BM
from verity.ml.circuits.ops.boolean import scalar as SC
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import dsa as D
from verity.ml.definitions.registry import dsv4_qnorm_insert as Q
from verity.ml.definitions.registry.boolean_attention import _Bits
from verity.ml.definitions.registry.boolean_dsa import F32CeilFtzV2, I32SubV2, U32AndV2, flat
from verity.ml.definitions.registry.boolean_flashmla import BIT_T, E8, F32, NvExp2fV2, NvLog2fV2

#: the stages of the right shift by 31 - k, largest first, each with the low bits its later stages read
_STAGES = ((16, 23), (8, 15), (4, 11), (2, 9), (1, 8))


def f32_to_u32_rz_byte(C, x):
    """`dsv4_qnorm_insert.f32_to_u32_rz_byte`: with k = e - 127 (no borrow: x >= 1 in magnitude), x < 2^32 iff k < 32, and then the
    truncation is (1.m << 8) >> (31 - k), 31 - k the NOT of k's five low bits; 0xFF from k >= 32 (+inf too), 0 for a NaN, a
    negative or k < 0."""
    m, e, s = x[:23], x[23:31], x[31]
    k, below = sub(C, e, const_bits(127, 8), 8)
    big = C.AND(NOT(below), or_reduce(C, k[5:]))
    nan = C.AND(_all(C, e), or_reduce(C, m))
    w = [ZERO] * 8 + list(m) + [ONE]
    for (sh, keep), r in zip(_STAGES, (NOT(k[4]), NOT(k[3]), NOT(k[2]), NOT(k[1]), NOT(k[0]))):
        w = _stage(C, r, w, sh, keep)
    ok = C.AND(NOT(s), NOT(nan))
    sat, fits = C.AND(ok, big), C.AND(ok, C.AND(NOT(below), NOT(big)))
    return [sat ^ C.AND(fits, b) for b in w[:8]]


def _stage(C, r, w, sh, keep):
    """`mux(r, w >> sh, w)` to ``keep`` bits, each r AND v made once: in the first stage a bit read shifted at i and unshifted at
    i + sh (its other operand zero) would be ANDed with r twice in a lowering that does not share terms."""
    ands = {}
    out = []
    for i in range(keep):
        v = (w[i + sh] if i + sh < len(w) else ZERO) ^ w[i]
        if v.key() not in ands:
            ands[v.key()] = C.AND(r, v)
        out.append(w[i] ^ ands[v.key()])
    return out


def _all(C, bits):
    out = bits[0]
    for b in bits[1:]:
        out = C.AND(out, b)
    return out


F32ToU32RzByteV2 = definition("F32ToU32RzByte", 2, f32_to_u32_rz_byte, (("x", F32),), E8, word=Q.F32ToU32RzByte,
                              doc="the low byte of cvt.rzi.u32.f32 on bits: F32ToU32RzByte_v1")

BITS = dict(wide=BC.Bf16ToF32, cvt=BG.F2fpBf16, add=EW.F32AddV4, mul=EW.F32MulV4, fma=EW.F32FmaV4, div=EW.F32DivV4,
            rsqrt=BM.RsqrtApprox, fabs=SC.F32Fabs, fmax=SC.F32Fmaxf, fmin=SC.F32Fminf, e4m3=SC.F32ToE4m3Sat, iadd=EW.I32Add,
            iand=U32AndV2, isub=I32SubV2, ceil=F32CeilFtzV2, byte=F32ToU32RzByteV2, log2=NvLog2fV2, exp2=NvExp2fV2)


def bit_ops(B) -> Q.Ops:
    """`dsv4_qnorm_insert.Ops` over `BITS`."""
    c = _Bits(B)
    return Q.Ops(B, BITS, lambda v: c.word(v, 32), lambda x: Coll(F32, concat([x.refs.slice(0, 31), B.call(BL.NOT, x[31]).refs])),
                 lambda v: c.word(v, 16), BIT_T, 16, lambda v: c.word(v, 8))


def _q(B, S, q, cs):
    """`Dsv4QNormRope_v1`'s body on bits."""
    return Q.q_norm_rope(bit_ops(B), S, q, cs)


Dsv4QNormRopeV2 = flat("Dsv4QNormRope", 2, ["D", "DR", "EPS"], lambda S: Q._q_type(S, BIT_T), _q, Q.Dsv4QNormRope)


def _kv(B, S, kv, cs):
    """`Dsv4KvRopeQuant_v1`'s body on bits."""
    return Q.kv_rope_quant(bit_ops(B), S, kv, cs)


Dsv4KvRopeQuantV2 = flat("Dsv4KvRopeQuant", 2, ["DN", "DR", "QB"], lambda S: Q._kv_type(S, BIT_T), _kv, Q.Dsv4KvRopeQuant)


def _insert(B, S, q, kv, cs):
    """`Dsv4QNormRopeKvInsert_v1`'s body on bits over `Dsv4QNormRope_v2` and `Dsv4KvRopeQuant_v2`."""
    return Q.insert(bit_ops(B), S, bind(Dsv4QNormRopeV2, D=S.DN + S.DR, DR=S.DR, EPS=S.EPS),
                    bind(Dsv4KvRopeQuantV2, DN=S.DN, DR=S.DR, QB=S.QB), q, kv, cs)


Dsv4QNormRopeKvInsertV2 = flat("Dsv4QNormRopeKvInsert", 2, ["NH", "NHP", "DN", "DR", "QB", "EPS"],
                               lambda S: Q._insert_type(S, BIT_T), _insert, Q.Dsv4QNormRopeKvInsert)


def _head(B, S, q, cs, w):
    """`Dsv4IndexerQHead_v1`'s body on bits."""
    return Q.indexer_q_head(bit_ops(B), S, q, cs, w)


Dsv4IndexerQHeadV2 = flat("Dsv4IndexerQHead", 2, ["D", "DR", "SCALE"], lambda S: Q._head_type(S, BIT_T), _head,
                          Q.Dsv4IndexerQHead)


def _token(B, S, q, cs, w):
    """`Dsv4IndexerQ_v1`'s body on bits over `Dsv4IndexerQHead_v2`."""
    return D.heads(B, bind(Dsv4IndexerQHeadV2, D=S.D, DR=S.DR, SCALE=S.SCALE), S.H, S.D, q, cs, w, E8, F32)


Dsv4IndexerQV2 = flat("Dsv4IndexerQ", 2, ["H", "D", "DR", "SCALE"], lambda S: Q._token_type(S, BIT_T), _token, Q.Dsv4IndexerQ)

BOOLEAN_DSV4_QNORM_INSERT = [F32ToU32RzByteV2, Dsv4QNormRopeV2, Dsv4KvRopeQuantV2, Dsv4QNormRopeKvInsertV2, Dsv4IndexerQHeadV2,
                             Dsv4IndexerQV2]
