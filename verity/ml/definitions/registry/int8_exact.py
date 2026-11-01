"""The exact-int8 profile's linear and attention (`TargetProfile.int8_exact`) as Definitions: what `verity_vllm.engine.int8_exact`
(the plain int8 linear) and `verity_vllm.engine.int8_attention` (compute accounting's integer LUT softmax) compute in torch, on the
CPU and the GPU alike.  The linear's product is `verity.ml.circuits.ops.gemm.ScaledMmS8_v1` over the sm_120 s8 IMMA step; the
attention's matmuls are the shared s8 family's plain arm (`verity.ml.circuits.ops.s8.S8Matmul_v1`: NOISE "none", no tail, COMMIT
"final"); every rounding is its own primitive, so the census counts each.

RN32 / RN64 are float32 / float64 round to nearest even, one rounding per operation; every division is a true IEEE division.

The quantizer (`Int8QuantElement`, one code; `Int8RowQuant{K}`, one scale over K values; `Int8StaticQuant{K}`, a given scale):

    s    = RN32(max(max_k |x_k|, RN32(1e-30)) / 127)            the max is exact, so its order is not part of the value
    X_k  = s8(clamp(rint(RN32(x_k / s)), -127, 127))           F32Div, then F32Rint (ties to even), then the exact clamp and cast

`Int8RowQuant_v3{K,R}` is `Int8RowQuant_v1{K}` with x as R rows of K/R values in the same leaf order, so a cut can read its port
as R rows where v1's is one.

The linear (`Int8Linear{K,N}` on one token row: `Int8RowQuant`, then `ScaledMmS8_v1{K,N,DOT=BlackwellS8ImmaDot32_v1}`, two Calls
in a served Program), weights W [N, K] s8 with per-row scales s_w; per output, `ScaledMmS8Coordinate_v1`:

    Y_n  = sum_k X_k W_nk                                      K/32 satfinite IMMA steps from 0, exact: |Y_n| <= 127 * 128 * K < 2^31
    y_n  = bf16(RN32(RN32(RN32(Y_n) s_a) s_w[n]))              S32ToF32Rn, F32MulV2 twice (the activation scale first), F2fpBf16

The attention, per request and KV head (`Int8KvQuant{S,D}`: its window of S keys) and per (query row, head) (`Int8LutAttentionRow{D,S,
SCALE}`: the row at position pos = S - 1 over its own S keys, the first S rows of its KV head's `Int8KvQuant`):

    Q'   = Int8RowQuant{D}(q), scale s_q                       one scale per (head, token)
    K'   = Int8StaticQuant{D}(k, s_k)                          ONE scale per (layer, KV head) (`engine.int8_k_scales`)
    V'   = Int8RowQuant{S}(v[:, c]) per channel c, scale s_v[c] over the whole window (`Int8ColumnQuant{S,D}`)
    S_j  = Q' . K'_j                                           S8Matmul_v1{S, D}
    A    = min(rint(RN64(RN64(RN64(RN64(s_q s_k) SCALE) log2 e) 16) 2^32), 2^32 - 1)    `Int8AttnScale{SCALE}`: fattn.cu's uint32_t fa_A
    m    = max_j S_j;  t_j = min(((m - S_j) A) >> 32, 255);  e_j = LUT[t_j],  LUT[t] = rint(127 2^(-t/16))     `Int8LutSoftmax{S}`
    L    = sum_j e_j;  r = RN32(1 / L);  z = RN32(e_0 / L);  P = (0, e_1, .., e_pos)      key 0 is its own sink block
    Y_c  = P . V'[c]                                           S8Matmul_v1{D, S'}, keys zero-padded to S' (a multiple of 32)
    out  = bf16(RN32(RN32(RN32(RN32(Y_c) r) + RN32(z V'_0c)) s_v[c]))      `Int8AttnOut`: apv_ref.py's served f, no dither

s_q s_k is exact in float64 (`F32MulWide`), and A is one 32-bit word: a value several proof blocks read is at most 32 bits.  What
every row of a KV group reads (K', V', s_v, f32(V'_0)) is computed once, by `Int8KvQuant`.

The primitives here are the ones no registry had: `F32Rint`, `F32ToS8Rzi`, `S8ToF32`, `F32MulWide`, `F64Mul`, `F64ToU32RniSat`,
`LutIndex`, `Exp2Lut127` (the LUT softmax's 256-entry gather, a u8 index to its 32-bit entry), `U8ToU16`.  Each is total on encodings; the docstrings say where the served path's domain is narrower.

The table is a value the program brings (PROTOCOL.md §5.3): `Exp2Lut127_v2`, `Int8LutSoftmax_v2{S}` and
`Int8LutAttentionRow_v2{D,S,SCALE}` are the v1 forms with LUT read from the registered `exp2_lut127` (`registry.tables`,
`TableRead8x2_v1`), which a `registered_values` derivation binds.
"""
from __future__ import annotations

import math
import struct

import numpy as np

from verity.ml.circuits.defs import bind, composite, primitive
from verity.ml.circuits.ops import gemm as GEMM
from verity.ml.circuits.ops import prims as CP
from verity.ml.circuits.ops import s8 as S8
from verity.ml.circuits.ops.scalar import F32Fabs, F32Fmaxf, F32Fminf, I32Add, I32Le, SelectI32
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.definitions.evidence import EVIDENCE
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry import tables as T
from verity.ml.definitions.registry.dsa import I32Sub
from verity.ml.definitions.registry.prims import BF16, F32, I32, bits_of, f32_of
from verity.ml.definitions.registry.topp_word_gates import I32ToF32

V8, F64 = Value(8), Value(64)
TOP = 127
SCALE_FLOOR = 1e-30
LOG2E = 1.4426950408889634
LUT_SIZE = 256
#: fattn_ref.py's table: LUT[t] = rint(127 2^(-t/16))
LUT = tuple(int(v) for v in np.rint(127.0 * np.exp2(-np.arange(LUT_SIZE) / 16.0)))
#: fattn.cu's A is a uint32_t: past it, A saturates
A_CAP = (1 << 32) - 1
#: the most padded keys P V' runs over in its window (`Accounting.Guarantees.Gates.LutPvWindow`, 32·S <= 132,096): P in [0, 127]
#: against V' in [-128, 127] keeps every product in [-16256, 16129], so 4,128 steps stay in s32 where 2^17 does not
PV_KEY_LIMIT = 132_096

_CONF = ("exact-model-tested; every Definition here against the torch of int8_exact and "
         "int8_attention (the torch the profile serves) on random inputs on the CPU, all words equal; that torch is bit-equal on the "
         f"RTX PRO 6000 to its CPU run and to fattn_ref.py ({EVIDENCE['int8_fattn_ref_check']['run_record']}: 5,087,232 linear words, "
         "884,736 attention words)")


def _f64(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits & 0xFFFFFFFFFFFFFFFF))[0]


def _f64_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _s32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


# ---- primitives ------------------------------------------------------------------------------------------------------------


@primitive("F32Rint", 1, [("x", F32)], F32, conformance=_CONF)
def F32Rint(x: int) -> int:
    """`cvt.rni.f32.f32` (torch.round): x rounded to the nearest integer, ties to even, as f32; NaN, infinities and zeros are
    themselves."""
    return bits_of(np.rint(f32_of(x)))


@primitive("F32ToS8Rzi", 1, [("x", F32)], V8, conformance=_CONF)
def F32ToS8Rzi(x: int) -> int:
    """`cvt.rzi.sat.s8.f32`: x truncated toward zero and saturated to [-128, 127], as an s8 code; NaN is 0.  Exact on the
    integers in [-127, 127] the quantizer's clamp leaves it."""
    v = float(f32_of(x))
    if math.isnan(v):
        return 0
    return max(-128, min(127, int(v) if math.isfinite(v) else (127 if v > 0 else -128))) & 0xFF


@primitive("S8ToF32", 1, [("c", V8)], F32, conformance=_CONF)
def S8ToF32(c: int) -> int:
    """The s8 code c as f32, exact."""
    return bits_of(np.float32(((c & 0xFF) ^ 0x80) - 0x80))


@primitive("F32MulWide", 1, [("a", F32), ("b", F32)], F64, conformance=_CONF)
def F32MulWide(a: int, b: int) -> int:
    """The float64 product of two float32 values, `mul.rn.f64(cvt.f64.f32(a), cvt.f64.f32(b))`: exact, since two 24-bit significands
    make at most 48 bits and the exponents stay in float64's normal range."""
    return _f64_bits(float(f32_of(a)) * float(f32_of(b)))


@primitive("F64Mul", 1, [("a", F64), ("b", F64)], F64, conformance=_CONF)
def F64Mul(a: int, b: int) -> int:
    """`mul.rn.f64`: RN64(a b)."""
    return _f64_bits(_f64(a) * _f64(b))


@primitive("F64ToU32RniSat", 1, [("x", F64)], I32, conformance=_CONF)
def F64ToU32RniSat(x: int) -> int:
    """A from its float64 value: min(rint(x), 2^32 - 1) (ties to even) as an unsigned 32-bit word, and 0 for a NaN or x <= 0.  The
    served x is positive (s_q, s_k and SCALE are), where this is torch's ``round(x).clamp(max=2^32 - 1).long()``."""
    v = _f64(x)
    if math.isnan(v) or v <= 0:
        return 0
    return int(min(np.rint(v), float(A_CAP)))


@primitive("LutIndex", 1, [("d", I32), ("a", I32)], V8, conformance=_CONF)
def LutIndex(d: int, a: int) -> int:
    """t = min(floor(max(d, 0) a / 2^32), 255) as an unsigned byte, d a signed 32-bit word and a an unsigned one: fattn's
    min(umulhi(m - S, A), 255).  The served d = m - S is in [0, 2^23), so torch's int64 product does not wrap."""
    return min((max(_s32(d), 0) * (a & 0xFFFFFFFF)) >> 32, LUT_SIZE - 1)


@primitive("Exp2Lut127", 1, [("t", V8)], I32, conformance=_CONF, tables=(("exp2_lut127", "registered"),))
def Exp2Lut127(t: int) -> int:
    """The LUT softmax's gather (`engine.int8_attention.lut_e`'s ``LUT[t]``): LUT[t] = rint(127 2^(-t/16)) for the unsigned byte t,
    a 256-entry table whose entries are in [0, 127], as a 32-bit word.  The table is fattn's, not the library's, so the primitive
    holds it as the program's registered `exp2_lut127` (`verity.ml.circuits.operations.program_data` lists it)."""
    return LUT[t & 0xFF]


@primitive("U8ToU16", 1, [("t", V8)], Value(16), conformance=_CONF)
def U8ToU16(t: int) -> int:
    """The byte t zero-extended to a 16-bit word: `TableRead8x2_v1`'s index."""
    return t & 0xFF


EXP2_LUT127_TABLE = T.table_type("exp2_lut127")


def exp2_lut127_words() -> list[int]:
    """The registered `exp2_lut127`'s words in leaf order: LUT[t] as a 32-bit entry, its low word first."""
    return T.table_words("exp2_lut127", lambda t: LUT[t])


@composite("Exp2Lut127", 2, [], lambda S: ((("t", V8), ("exp2_lut127", EXP2_LUT127_TABLE)), I32),
           conformance="exact-model-tested: Exp2Lut127_v1 with LUT[t] read from the registered table")
def Exp2Lut127V2(B, S, t, exp2_lut127):
    """`Exp2Lut127_v1` with LUT[t] read from the registered `exp2_lut127` (`TableRead8x2_v1` at t zero-extended, `U8ToU16_v1`)
    rather than held by the evaluator."""
    return B.call(T.table_read(8, 2), B.call(U8ToU16, t), exp2_lut127)


C_TOP = CP.const(32, bits_of(float(TOP)))
C_NEG_TOP = CP.const(32, bits_of(-float(TOP)))
C_FLOOR = CP.const(32, bits_of(SCALE_FLOOR))
C_ONE = CP.const(32, bits_of(1.0))
C_LOG2E = CP.const(64, _f64_bits(LOG2E))
C_16 = CP.const(64, _f64_bits(16.0))
C_2_32 = CP.const(64, _f64_bits(4294967296.0))
ZERO8 = CP.const(8, 0)

INT8_PRIMITIVES = [F32Rint, F32ToS8Rzi, S8ToF32, F32MulWide, F64Mul, F64ToU32RniSat, LutIndex, Exp2Lut127, U8ToU16]


def plain_arm(K: int, N: int, A: tuple[int, int] = (-TOP, TOP)) -> dict:
    """The s8 family's plain serving arm (`S8.SMOKE_PLAIN`) at K, N, with the left operand's interval A."""
    return S8.SMOKE_PLAIN | {"K": K, "L": K, "N": N, "A": A}


def padded(n: int) -> int:
    """n rounded up to the s8 family's k step (32)."""
    return -(-n // 32) * 32


# ---- the quantizer -------------------------------------------------------------------------------------------------------


@composite("Int8QuantElement", 1, [], lambda S: ((("x", BF16), ("s", F32)), V8), conformance=_CONF)
def Int8QuantElement(B, S, x, s):
    """One code: s8(clamp(rint(RN32(f32(x) / s)), -127, 127))."""
    r = B.call(F32Rint, B.call(P.F32Div, B.call(P.Bf16ToF32, x), s))
    return B.call(F32ToS8Rzi, B.call(F32Fminf, B.call(F32Fmaxf, r, B.call(C_NEG_TOP)), B.call(C_TOP)))


@composite("Int8RowScale", 1, ["K"], lambda S: ((("x", Array(S.K, BF16)),), F32), conformance=_CONF)
def Int8RowScale(B, S, x):
    """s = RN32(max(max_k |f32(x_k)|, RN32(1e-30)) / 127): an fmaxf fold from +0 (an exact max) and one division."""
    a = B.call(CP.ZERO32)
    for k in range(S.K):
        a = B.call(F32Fmaxf, a, B.call(F32Fabs, B.call(P.Bf16ToF32, x[k])))
    return B.call(P.F32Div, B.call(F32Fmaxf, a, B.call(C_FLOOR)), B.call(C_TOP))


@composite("Int8StaticQuant", 1, ["K"], lambda S: ((("x", Array(S.K, BF16)), ("s", F32)), Array(S.K, V8)), conformance=_CONF)
def Int8StaticQuant(B, S, x, s):
    """K codes under a given scale s (`Int8QuantElement` each)."""
    return B.batch(Int8QuantElement, x, s, axes=(0, None))


def _row_quant(B, K: int, x: Coll):
    s = B.call(bind(Int8RowScale, K=K), x)
    return tuple_of(B.call(bind(Int8StaticQuant, K=K), x, s), s)


@composite("Int8RowQuant", 1, ["K"], lambda S: ((("x", Array(S.K, BF16)),), Tuple(Array(S.K, V8), F32)), conformance=_CONF)
def Int8RowQuant(B, S, x):
    """(codes, s): the row's scale (`Int8RowScale`) and its K codes under it (`Int8StaticQuant`)."""
    return _row_quant(B, S.K, x)


def row_parts_refusal(b: dict) -> str | None:
    """`bind`'s refusal for `Int8RowQuant_v3`: x is R rows of K/R values, so K and R are positive integers and R divides K."""
    K, R = b.get("K"), b.get("R")
    for name, v in (("K", K), ("R", R)):
        if v is not None and (type(v) is not int or v < 1):
            return f"{name}={v!r}: x is R rows of K/R values, so K and R are positive integers"
    if K is not None and R is not None and K % R:
        return f"K={K}, R={R}: x is R rows of K/R values, so R divides K"
    return None


@composite("Int8RowQuant", 3, ["K", "R"], lambda S: ((("x", Array(S.R, Array(S.K // S.R, BF16))),), Tuple(Array(S.K, V8), F32)),
           conformance="exact-model-tested: Int8RowQuant_v1{K} with x read as R rows of K/R values", refuse=row_parts_refusal)
def Int8RowQuantV3(B, S, x):
    """`Int8RowQuant_v1{K}` with x as R rows of K/R values in v1's leaf order, so a cut can read the port as R rows: v1's body over
    x viewed as one array of K by reference, with no gate of its own."""
    return _row_quant(B, S.K, Coll(Array(S.K, BF16), x.refs))


# ---- the linear ----------------------------------------------------------------------------------------------------------


def scaled_mm(K: int, N: int):
    """`ScaledMmS8_v1{K, N, DOT=BlackwellS8ImmaDot32_v1}` (`verity.ml.circuits.ops.gemm`): one quantized token row against the N
    weight rows, `ScaledMmS8Coordinate_v1` per output, each a `Q_word` unit (`verity.ml.circuits.cut.separable`)."""
    return bind(GEMM.ScaledMmS8, K=K, N=N, **GEMM.GEMM_SM120_S8)


@composite("Int8Linear", 1, ["K", "N"],
           lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, V8))), ("sw", Array(S.N, F32))),
                      Tuple(Array(S.K, V8), F32, Array(S.N, BF16))), conformance=_CONF)
def Int8Linear(B, S, x, w, sw):
    """The plain int8 linear on one token row: (X, s_a, y), `Int8RowQuant` then `ScaledMmS8_v1` (`scaled_mm`)."""
    q = B.call(bind(Int8RowQuant, K=S.K), x)
    return tuple_of(q[0], q[1], B.call(scaled_mm(S.K, S.N), q[0], q[1], w, sw))


# ---- the attention -------------------------------------------------------------------------------------------------------


@composite("Int8AttnScale", 1, ["SCALE"], lambda S: ((("sq", F32), ("sk", F32)), I32), conformance=_CONF)
def Int8AttnScale(B, S, sq, sk):
    """A = min(rint(RN64(RN64(RN64(RN64(s_q s_k) SCALE) log2 e) 16) 2^32), 2^32 - 1), left to right."""
    x = B.call(F32MulWide, sq, sk)
    for c in (CP.const(64, _f64_bits(float(S.SCALE))), C_LOG2E, C_16, C_2_32):
        x = B.call(F64Mul, x, B.call(c))
    return B.call(F64ToU32RniSat, x)


def _softmax_sig(S, table: bool = False):
    params = (("sc", Array(S.S, I32)), ("a", I32)) + ((("exp2_lut127", EXP2_LUT127_TABLE),) if table else ())
    return params, Tuple(Array(S.S, V8), F32, F32)


@composite("Int8LutSoftmax", 1, ["S"], _softmax_sig, conformance=_CONF)
def Int8LutSoftmax(B, S, sc, a):
    """(P, r, z) of one row's S scores under A: m = max S, e_j = LUT[LutIndex(m - S_j, A)], L = sum e, r = RN32(1 / L),
    z = RN32(e_0 / L), P = (0, e_1, .., e_{S-1}) as s8 codes."""
    return _softmax(B, S, sc, a, lambda t: B.call(Exp2Lut127, t))


@composite("Int8LutSoftmax", 2, ["S"], lambda S: _softmax_sig(S, table=True),
           conformance="exact-model-tested: Int8LutSoftmax_v1 with LUT[t] read from the registered table")
def Int8LutSoftmaxV2(B, S, sc, a, exp2_lut127):
    """`Int8LutSoftmax_v1` with each e_j read from the registered `exp2_lut127` (`Exp2Lut127_v2`)."""
    return _softmax(B, S, sc, a, lambda t: B.call(Exp2Lut127V2, t, exp2_lut127))


def _softmax(B, S, sc, a, exp2):
    m = sc[0]
    for j in range(1, S.S):
        m = B.call(SelectI32, B.call(I32Le, m, sc[j]), sc[j], m)
    e = [exp2(B.call(LutIndex, B.call(I32Sub, m, sc[j]), a)) for j in range(S.S)]
    total = e[0]
    for j in range(1, S.S):
        total = B.call(I32Add, total, e[j])
    lf = B.call(I32ToF32, total)
    r = B.call(P.F32Div, B.call(C_ONE), lf)
    z = B.call(P.F32Div, lf if S.S == 1 else B.call(I32ToF32, e[0]), lf)     # S = 1: L is e_0, converted once (Q_word)
    p = [B.call(ZERO8)] + [B.call(S8.S32LowByte, e[j]) for j in range(1, S.S)]
    return tuple_of(Coll(Array(S.S, V8), concat([c.refs for c in p])), r, z)


@composite("Int8AttnOut", 1, [], lambda S: ((("y", I32), ("r", F32), ("z", F32), ("v0", F32), ("sv", F32)), BF16), conformance=_CONF)
def Int8AttnOut(B, S, y, r, z, v0, sv):
    """bf16(RN32(RN32(RN32(RN32(y) r) + RN32(z v0)) s_v)), v0 = f32(V'_0) of the channel."""
    yr = B.call(P.F32Mul, B.call(I32ToF32, y), r)
    return B.call(P.F32ToBf16Rn, B.call(P.F32Mul, B.call(P.F32Add, yr, B.call(P.F32Mul, z, v0)), sv))


def _pad_rows(B, rows: list[Coll], n: int) -> Coll:
    """Rows of s8 codes, each zero-padded to `padded(n)` codes, as one Array<len(rows), Array<padded(n), V8>>."""
    zero = B.call(ZERO8).refs
    pad = padded(n) - n
    return Coll(Array(len(rows), Array(padded(n), V8)), concat([p for r in rows for p in [r.refs] + [zero] * pad]))


def _row_sig(S, table: bool = False):
    D, K = S.D, S.S
    return ((("q", Array(D, BF16)), ("kq", Array(K, Array(D, V8))), ("sk", F32), ("vq", Array(K, Array(D, V8))), ("sv", Array(D, F32)),
             ("v0", Array(D, F32))) + ((("exp2_lut127", EXP2_LUT127_TABLE),) if table else ()), Array(D, BF16))


def row_refusal(b: dict) -> str | None:
    """`bind`'s refusal for a Row: Q' K'^T steps over D, so D is a multiple of 32; P V' steps over the keys zero-padded to
    `padded(S)`, which stays within `PV_KEY_LIMIT` (132,096, P V' 's own window), past which its s8 sum can leave s32."""
    D, K = b.get("D"), b.get("S")
    if isinstance(D, int) and (D < 32 or D % 32):
        return f"D={D}: Q' K'^T steps 32 channels at a time, so the head dim is a positive multiple of 32"
    if isinstance(K, int) and K < 1:
        return f"S={K}: a row attends to at least its own key"
    if isinstance(K, int) and padded(K) > PV_KEY_LIMIT:
        return (f"S={K}: P V' over {padded(K)} padded keys is past its window of {PV_KEY_LIMIT} (LutPvWindow), where its s8 sum can "
                f"leave s32; a context of at most {PV_KEY_LIMIT} keys binds")
    return None


@composite("Int8LutAttentionRow", 1, ["D", "S", "SCALE"], _row_sig, conformance=_CONF, refuse=row_refusal)
def Int8LutAttentionRow(B, S, q, kq, sk, vq, sv, v0):
    """One query row over its S keys 0..pos (pos = S - 1): Q' (`Int8RowQuant`), the scores (`S8Matmul_v1{S, D}`), A
    (`Int8AttnScale`), the LUT softmax (`Int8LutSoftmax`), P V' (`S8Matmul_v1{D, S'}` over the keys zero-padded to a multiple of
    32, P in [0, 127]) and the output (`Int8AttnOut` per channel).  `kq` and `vq` are K' and V' by key (`Int8KvQuant`'s first S
    rows), `sk` the KV head's K scale, `sv` V's per-channel scales and `v0` f32(V'_0)."""
    return _row(B, S, q, kq, sk, vq, sv, v0, lambda sc, a: B.call(bind(Int8LutSoftmax, S=S.S), sc, a))


@composite("Int8LutAttentionRow", 2, ["D", "S", "SCALE"], lambda S: _row_sig(S, table=True),
           conformance="exact-model-tested: Int8LutAttentionRow_v1 with LUT[t] read from the registered table", refuse=row_refusal)
def Int8LutAttentionRowV2(B, S, q, kq, sk, vq, sv, v0, exp2_lut127):
    """`Int8LutAttentionRow_v1` whose softmax reads the registered `exp2_lut127` (`Int8LutSoftmax_v2`)."""
    return _row(B, S, q, kq, sk, vq, sv, v0, lambda sc, a: B.call(bind(Int8LutSoftmaxV2, S=S.S), sc, a, exp2_lut127))


def _row(B, S, q, kq, sk, vq, sv, v0, softmax):
    D, K = S.D, S.S
    qq = B.call(bind(Int8RowQuant, K=D), q)
    sc = B.call(bind(S8.S8Matmul, **plain_arm(D, K)), qq[0], kq)
    soft = softmax(sc, B.call(bind(Int8AttnScale, SCALE=S.SCALE), qq[1], sk))
    p = _pad_rows(B, [soft[0]], K)[0]
    v = _pad_rows(B, [Coll(Array(K, V8), concat([vq[j][c].refs for j in range(K)])) for c in range(D)], K)
    y = B.call(bind(S8.S8Matmul, **plain_arm(padded(K), D, (0, TOP))), p, v)
    return B.batch(Int8AttnOut, y, soft[1], soft[2], v0, sv, axes=(0, None, None, 0, 0))


@composite("Int8ColumnQuant", 1, ["S", "D"],
           lambda S: ((("v", Array(S.S, Array(S.D, BF16))),), Tuple(Array(S.D, Array(S.S, V8)), Array(S.D, F32))), conformance=_CONF)
def Int8ColumnQuant(B, S, v):
    """V' by channel and its scales: `Int8RowQuant{S}` on each of the D columns of the S x D window."""
    K, D = S.S, S.D
    cols = Coll(Array(D, Array(K, BF16)), concat([v[j][c].refs for c in range(D) for j in range(K)]))
    out = B.batch(bind(Int8RowQuant, K=K), cols, axes=(0,))
    return tuple_of(Coll(Array(D, Array(K, V8)), concat([out[c][0].refs for c in range(D)])),
                    Coll(Array(D, F32), concat([out[c][1].refs for c in range(D)])))


def _kv_sig(S):
    K, D = S.S, S.D
    kv = Array(K, Array(D, BF16))
    return (("k", kv), ("v", kv), ("sk", F32)), Tuple(Array(K, Array(D, V8)), Array(K, Array(D, V8)), Array(D, F32), Array(D, F32))


@composite("Int8KvQuant", 1, ["S", "D"], _kv_sig, conformance=_CONF)
def Int8KvQuant(B, S, k, v, sk):
    """One KV head's window of S keys (K after RoPE and V, bf16) under its K scale `sk`: (K', V' by key, s_v, f32(V'_0)).  K' per
    key (`Int8StaticQuant`), V' per channel over the window (`Int8ColumnQuant`), and its first row widened (`S8ToF32`): what every
    query row of the KV head reads (`Int8LutAttentionRow`), computed once."""
    K, D = S.S, S.D
    kq = B.batch(bind(Int8StaticQuant, K=D), k, sk, axes=(0, None))
    cq = B.call(bind(Int8ColumnQuant, S=K, D=D), v)
    vq = Coll(Array(K, Array(D, V8)), concat([cq[0][c][j].refs for j in range(K) for c in range(D)]))
    v0 = B.batch(S8ToF32, vq[0], axes=(0,))
    return tuple_of(kq, vq, cq[1], v0)


INT8_COMPOSITES = [Int8QuantElement, Int8RowScale, Int8StaticQuant, Int8RowQuant, Int8Linear, Int8AttnScale, Int8LutSoftmax,
                   Int8AttnOut, Int8LutAttentionRow, Int8ColumnQuant, Int8KvQuant, Exp2Lut127V2, Int8LutSoftmaxV2, Int8LutAttentionRowV2,
                   Int8RowQuantV3]
