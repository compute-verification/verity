"""The exact-int8 profile's linear and attention (`TargetProfile.int8_exact`) as Definitions: what `verity_vllm.engine.int8_exact`
(the plain int8 linear) and `verity_vllm.engine.int8_attention` (compute accounting's integer LUT softmax) compute in torch, on the
CPU and the GPU alike.  Every matmul is the shared s8 family's plain arm (`verity.ml.circuits.ops.s8.S8Matmul_v1`: NOISE "none",
no tail, COMMIT "final"); every rounding is its own primitive, so the census counts each.

RN32 / RN64 are float32 / float64 round to nearest even, one rounding per operation; every division is a true IEEE division.

The quantizer (`Int8QuantElement`, one code; `Int8RowQuant{K}`, one scale over K values; `Int8StaticQuant{K}`, a given scale):

    s    = RN32(max(max_k |x_k|, RN32(1e-30)) / 127)            the max is exact, so its order is not part of the value
    X_k  = s8(clamp(rint(RN32(x_k / s)), -127, 127))           F32Div, then F32Rint (ties to even), then the exact clamp and cast

The linear (`Int8Linear{K,N}` on one token row: `Int8RowQuant`, then `Int8ScaledMm`), weights W [N, K] s8 with per-row scales s_w:

    Y_n  = sum_k X_k W_nk                                      S8Matmul_v1{N, K}, exact in s32 (`s8.no_wrap`)
    y_n  = bf16(RN32(RN32(RN32(Y_n) s_a) s_w[n]))              `Int8Dequant`: the activation scale first

The attention (`Int8LutAttentionWindow{N,S,H,G,D,SCALE}`: one request's window of S keys and its last N query rows, H query heads over
G KV heads; `Int8LutAttentionRow{D,S,SCALE}` one query row at position pos = S - 1 of its own S keys):

    Q'   = Int8RowQuant{D}(q), scale s_q                       one scale per (head, token)
    K'   = Int8StaticQuant{D}(k, s_k)                          ONE scale per (layer, KV head) (`engine.int8_k_scales`)
    V'   = Int8RowQuant{S}(v[:, c]) per channel c, scale s_v[c] over the whole window (`Int8ColumnQuant{S,D}`)
    S_j  = Q' . K'_j                                           S8Matmul_v1{S, D}
    A    = min(rint(RN64(RN64(RN64(RN64(s_q s_k) SCALE) log2 e) 16) 2^32), 2^32 - 1)    `Int8AttnScale{SCALE}`: fattn.cu's uint32_t fa_A
    m    = max_j S_j;  t_j = min(((m - S_j) A) >> 32, 255);  e_j = LUT[t_j],  LUT[t] = rint(127 2^(-t/16))     `Int8LutSoftmax{S}`
    L    = sum_j e_j;  r = RN32(1 / L);  z = RN32(e_0 / L);  P = (0, e_1, .., e_pos)      key 0 is its own sink block
    Y_c  = P . V'[c]                                           S8Matmul_v1{D, S'}, keys zero-padded to S' (a multiple of 32)
    out  = bf16(RN32(RN32(RN32(RN32(Y_c) r) + RN32(z V'_0c)) s_v[c]))      `Int8AttnOut`: apv_ref.py's served f, no dither

s_q s_k is exact in float64 (`F32MulWide`), and A is one 32-bit word: a value several proof units read is at most 32 bits.  What
every row of a KV group reads (K', V', s_v, f32(V'_0)) is computed once, by the window.

The primitives here are the ones no registry had: `F32Rint`, `F32ToS8Rzi`, `S8ToF32`, `F32MulWide`, `F64Mul`, `F64ToU32RniSat`,
`LutIndex`, `Exp2Lut127`.  Each is total on encodings; the docstrings say where the served path's domain is narrower.
"""
from __future__ import annotations

import math
import struct

import numpy as np

from verity.ml.circuits.defs import bind, composite, primitive
from verity.ml.circuits.ops import prims as CP
from verity.ml.circuits.ops import s8 as S8
from verity.ml.circuits.ops.scalar import F32Fabs, F32Fmaxf, F32Fminf, I32Add, I32Le, SelectI32
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.dsa import I32Sub
from verity_vllm.program.registry.prims import BF16, F32, I32, bits_of, f32_of
from verity_vllm.program.registry.topp_word_gates import I32ToF32

V8, F64 = Value(8), Value(64)
TOP = 127
SCALE_FLOOR = 1e-30
LOG2E = 1.4426950408889634
LUT_SIZE = 256
#: fattn_ref.py's table: LUT[t] = rint(127 2^(-t/16))
LUT = tuple(int(v) for v in np.rint(127.0 * np.exp2(-np.arange(LUT_SIZE) / 16.0)))
#: fattn.cu's A is a uint32_t: past it, A saturates
A_CAP = (1 << 32) - 1

_CONF = ("exact-model-tested; tests/program/test_int8_exact_defs.py: every Definition here against verity_vllm.engine.int8_exact / "
         "int8_attention (the torch the profile serves) on random inputs on the CPU, all words equal; that torch is bit-equal on the "
         "RTX PRO 6000 to its CPU run and to fattn_ref.py (run r20261017-211407-3e75: 5,087,232 linear words, 884,736 attention words)")


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


@primitive("LutIndex", 1, [("d", I32), ("a", I32)], I32, conformance=_CONF)
def LutIndex(d: int, a: int) -> int:
    """t = min(floor(max(d, 0) a / 2^32), 255), d a signed 32-bit word and a an unsigned one: fattn's min(umulhi(m - S, A), 255).
    The served d = m - S is in [0, 2^23), so torch's int64 product does not wrap."""
    return min((max(_s32(d), 0) * (a & 0xFFFFFFFF)) >> 32, LUT_SIZE - 1)


@primitive("Exp2Lut127", 1, [("t", I32)], I32, conformance=_CONF)
def Exp2Lut127(t: int) -> int:
    """LUT[t] = rint(127 2^(-t/16)), the signed t clamped to [0, 255] (`LutIndex`'s range already)."""
    return LUT[min(max(_s32(t), 0), LUT_SIZE - 1)]


C_TOP = CP.const(32, bits_of(float(TOP)))
C_NEG_TOP = CP.const(32, bits_of(-float(TOP)))
C_FLOOR = CP.const(32, bits_of(SCALE_FLOOR))
C_ONE = CP.const(32, bits_of(1.0))
C_LOG2E = CP.const(64, _f64_bits(LOG2E))
C_16 = CP.const(64, _f64_bits(16.0))
C_2_32 = CP.const(64, _f64_bits(4294967296.0))
ZERO8 = CP.const(8, 0)

INT8_PRIMITIVES = [F32Rint, F32ToS8Rzi, S8ToF32, F32MulWide, F64Mul, F64ToU32RniSat, LutIndex, Exp2Lut127]


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


@composite("Int8RowQuant", 1, ["K"], lambda S: ((("x", Array(S.K, BF16)),), Tuple(Array(S.K, V8), F32)), conformance=_CONF)
def Int8RowQuant(B, S, x):
    """(codes, s): the row's scale (`Int8RowScale`) and its K codes under it (`Int8StaticQuant`)."""
    s = B.call(bind(Int8RowScale, K=S.K), x)
    return tuple_of(B.call(bind(Int8StaticQuant, K=S.K), x, s), s)


# ---- the linear ----------------------------------------------------------------------------------------------------------


@composite("Int8Dequant", 1, [], lambda S: ((("y", I32), ("sa", F32), ("sw", F32)), BF16), conformance=_CONF)
def Int8Dequant(B, S, y, sa, sw):
    """bf16(RN32(RN32(RN32(y) s_a) s_w))."""
    return B.call(P.F32ToBf16Rn, B.call(P.F32Mul, B.call(P.F32Mul, B.call(I32ToF32, y), sa), sw))


@composite("Int8ScaledDot", 1, ["K"], lambda S: ((("xq", Array(S.K, V8)), ("sa", F32), ("w", Array(S.K, V8)), ("sw", F32)), BF16),
           conformance=_CONF)
def Int8ScaledDot(B, S, xq, sa, w, sw):
    """One output: `S8Matmul_v1{N=1, K}`'s plain arm against one weight row, then `Int8Dequant`."""
    y = B.call(bind(S8.S8Matmul, **plain_arm(S.K, 1)), xq, w.reshape(Array(1, Array(S.K, V8))))
    return B.call(Int8Dequant, y[0], sa, sw)


def _mm_sig(S):
    return (("xq", Array(S.K, V8)), ("sa", F32), ("w", Array(S.N, Array(S.K, V8))), ("sw", Array(S.N, F32))), Array(S.N, BF16)


@composite("Int8ScaledMm", 1, ["K", "N"], _mm_sig, conformance=_CONF)
def Int8ScaledMm(B, S, xq, sa, w, sw):
    """One quantized row against the N weight rows, `Int8ScaledDot` per output: the N members read only the parameters, so the
    query of record cuts one member (`verity.ml.circuits.cut.separable`) rather than the whole fan-in."""
    return B.batch(bind(Int8ScaledDot, K=S.K), xq, sa, w, sw, axes=(None, None, 0, 0))


@composite("Int8Linear", 1, ["K", "N"],
           lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, V8))), ("sw", Array(S.N, F32))),
                      Tuple(Array(S.K, V8), F32, Array(S.N, BF16))), conformance=_CONF)
def Int8Linear(B, S, x, w, sw):
    """The plain int8 linear on one token row: (X, s_a, y), `Int8RowQuant` then `Int8ScaledMm`."""
    q = B.call(bind(Int8RowQuant, K=S.K), x)
    return tuple_of(q[0], q[1], B.call(bind(Int8ScaledMm, K=S.K, N=S.N), q[0], q[1], w, sw))


# ---- the attention -------------------------------------------------------------------------------------------------------


@composite("Int8AttnScale", 1, ["SCALE"], lambda S: ((("sq", F32), ("sk", F32)), I32), conformance=_CONF)
def Int8AttnScale(B, S, sq, sk):
    """A = min(rint(RN64(RN64(RN64(RN64(s_q s_k) SCALE) log2 e) 16) 2^32), 2^32 - 1), left to right."""
    x = B.call(F32MulWide, sq, sk)
    for c in (CP.const(64, _f64_bits(float(S.SCALE))), C_LOG2E, C_16, C_2_32):
        x = B.call(F64Mul, x, B.call(c))
    return B.call(F64ToU32RniSat, x)


@composite("Int8LutSoftmax", 1, ["S"], lambda S: ((("sc", Array(S.S, I32)), ("a", I32)), Tuple(Array(S.S, V8), F32, F32)),
           conformance=_CONF)
def Int8LutSoftmax(B, S, sc, a):
    """(P, r, z) of one row's S scores under A: m = max S, e_j = LUT[LutIndex(m - S_j, A)], L = sum e, r = RN32(1 / L),
    z = RN32(e_0 / L), P = (0, e_1, .., e_{S-1}) as s8 codes."""
    m = sc[0]
    for j in range(1, S.S):
        m = B.call(SelectI32, B.call(I32Le, m, sc[j]), sc[j], m)
    e = [B.call(Exp2Lut127, B.call(LutIndex, B.call(I32Sub, m, sc[j]), a)) for j in range(S.S)]
    total = e[0]
    for j in range(1, S.S):
        total = B.call(I32Add, total, e[j])
    lf = B.call(I32ToF32, total)
    r = B.call(P.F32Div, B.call(C_ONE), lf)
    z = B.call(P.F32Div, B.call(I32ToF32, e[0]), lf)
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


def _row_sig(S):
    D, K = S.D, S.S
    return ((("q", Array(D, BF16)), ("kq", Array(K, Array(D, V8))), ("sk", F32), ("vq", Array(D, Array(K, V8))), ("sv", Array(D, F32)),
             ("v0", Array(D, F32))), Array(D, BF16))


@composite("Int8LutAttentionRow", 1, ["D", "S", "SCALE"], _row_sig, conformance=_CONF)
def Int8LutAttentionRow(B, S, q, kq, sk, vq, sv, v0):
    """One query row over its S keys 0..pos (pos = S - 1): Q' (`Int8RowQuant`), the scores (`S8Matmul_v1{S, D}`), A
    (`Int8AttnScale`), the LUT softmax (`Int8LutSoftmax`), P V' (`S8Matmul_v1{D, S'}` over the keys zero-padded to a multiple of
    32, P in [0, 127]) and the output (`Int8AttnOut` per channel).  `vq` is V' by channel, `sv` its per-channel scales, `v0` f32(V'_0)."""
    D, K = S.D, S.S
    qq = B.call(bind(Int8RowQuant, K=D), q)
    sc = B.call(bind(S8.S8Matmul, **plain_arm(D, K)), qq[0], kq)
    soft = B.call(bind(Int8LutSoftmax, S=K), sc, B.call(bind(Int8AttnScale, SCALE=S.SCALE), qq[1], sk))
    p = _pad_rows(B, [soft[0]], K)[0]
    v = _pad_rows(B, [vq[c] for c in range(D)], K)
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


def _window_sig(S):
    N, K, H, G, D = S.N, S.S, S.H, S.G, S.D
    if H % G or not 0 < N <= K:
        raise ValueError(f"Int8LutAttentionWindow: H = {H} query heads over G = {G} KV heads, N = {N} rows of an S = {K}-key window")
    kv = Array(K, Array(G, Array(D, BF16)))
    return (("q", Array(N, Array(H, Array(D, BF16)))), ("k", kv), ("v", kv), ("ks", Array(G, F32))), Array(N, Array(H, Array(D, BF16)))


@composite("Int8LutAttentionWindow", 1, ["N", "S", "H", "G", "D", "SCALE"], _window_sig, conformance=_CONF)
def Int8LutAttentionWindow(B, S, q, k, v, ks):
    """One request's window: its S keys (K after RoPE and V, bf16) and its last N query rows (positions S - N .. S - 1), H query
    heads over G KV heads (head h reads KV head h // (H / G)), the layer's K scales `ks`.  Per KV head: K' per key
    (`Int8StaticQuant`), V' per channel over the window (`Int8ColumnQuant`) and f32(V'_0); per (row, head): `Int8LutAttentionRow` over the keys
    0..pos."""
    N, K, H, G, D = S.N, S.S, S.H, S.G, S.D
    rep = H // G
    out: dict[tuple[int, int], Coll] = {}
    for g in range(G):
        kg = Coll(Array(K, Array(D, BF16)), concat([k[j][g].refs for j in range(K)]))
        vg = Coll(Array(K, Array(D, BF16)), concat([v[j][g].refs for j in range(K)]))
        kq = B.batch(bind(Int8StaticQuant, K=D), kg, ks[g], axes=(0, None))
        vq = B.call(bind(Int8ColumnQuant, S=K, D=D), vg)
        v0 = B.batch(S8ToF32, Coll(Array(D, V8), concat([vq[0][c][0].refs for c in range(D)])), axes=(0,))
        for i in range(N):
            n = K - N + i + 1
            vn = Coll(Array(D, Array(n, V8)), concat([vq[0][c][0:n].refs for c in range(D)]))
            row = bind(Int8LutAttentionRow, D=D, S=n, SCALE=S.SCALE)
            for h in range(g * rep, (g + 1) * rep):
                out[(i, h)] = B.call(row, q[i][h], kq[0:n], ks[g], vn, vq[1], v0)
    return Coll(Array(N, Array(H, Array(D, BF16))), concat([out[(i, h)].refs for i in range(N) for h in range(H)]))


INT8_COMPOSITES = [Int8QuantElement, Int8RowScale, Int8StaticQuant, Int8RowQuant, Int8Dequant, Int8ScaledDot, Int8ScaledMm, Int8Linear,
                   Int8AttnScale, Int8LutSoftmax, Int8AttnOut, Int8LutAttentionRow, Int8ColumnQuant, Int8LutAttentionWindow]
