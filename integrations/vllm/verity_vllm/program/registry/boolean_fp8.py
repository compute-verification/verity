"""The served FP8 linear on bits: `fp8`'s composites, version 2, their v1 bodies line for line over Boolean Definitions, so each
one's word view is its v1 at the same statics.

    Fp8RowAbsMax_v2{K}, Fp8RowScale_v2, Fp8ActQuant_v2{K}                 the per-token activation quantisation
    DotE4m3_v2{K}, ScaledMmFp8Coordinate_v2{K,BIAS}, ScaledMmFp8_v2{K,N,BIAS}, QuantLinearFp8_v2{K,N,BIAS}
                                                                          the per-tensor cutlass GEMM (k32 wgmma chain, fused epilogue)
    Fp8GroupScale_v2{G}, Fp8GroupQuant_v2{K,G}, ScaledMmFp8BlockCoordinate_v2{K,G}, ScaledMmFp8Block_v2{K,N,G},
    ScaledMmFp8BlockCoordinateGivenScale_v2{K,G}, ScaledMmFp8BlockSharedScaleTile_v2{K,G}, ScaledMmFp8BlockSharedScale_v2{K,N,G}
                                                                          the block-scaled linear (128x128 weight blocks)

Every word is its bits (`verity.ir.boolean.word_type`): bf16 `Array<16, Value<1>>`, f32 `Array<32, Value<1>>`, e4m3
`Array<8, Value<1>>`. The callees are the Boolean versions of v1's: `verity.ml.boolean.scalar`'s f32 arithmetic, max / min and
casts, `verity.ml.boolean.gemm`'s `HopperE4m3QgmmaDot32_v2` and `F2fpBf16_v2`; the word constants (+0, 448, -448, the minimum
scale, the group quantiser's eps) are written with the constant bits.
"""
from __future__ import annotations

import numpy as np

from verity.ir import boolean as BL
from verity.ir.defs import bind, composite
from verity.ir.refs import Coll, concat, tuple_of
from verity.ir.types import Array, Tuple
from verity.ml.boolean import gemm as BG
from verity.ml.boolean import scalar as SC
from verity_vllm.program.registry import fp8 as F8
from verity_vllm.program.registry.boolean_attention import _Bits, _view
from verity_vllm.program.registry.prims import bits_of

BF16 = Array(16, BL.BIT)
F32 = Array(32, BL.BIT)
E4M3 = Array(8, BL.BIT)

F448 = bits_of(np.float32(448.0))
F_NEG448 = bits_of(np.float32(-448.0))
EPS_QUANT = bits_of(np.float32(1e-10))


@composite("Fp8RowAbsMax", 2, ["K"], lambda S: ((("x", Array(S.K, BF16)),), F32), conformance="exact-model-tested")
def Fp8RowAbsMaxV2(B, S, x):
    """`Fp8RowAbsMax_v1`'s body on bits: per-thread `fmaxf` folds from +0 over the 8-element vectors, a shuffle tree per warp,
    the 8 warp maxima folded in order."""
    K = S.K
    assert K % F8.QUANT_VEC == 0 and K >= F8.QUANT_THREADS, K
    c = _Bits(B)
    partials = []
    for t in range(F8.QUANT_THREADS):
        p = c.word(0, 32)
        for e in F8._thread_elements(K, t):
            p = B.call(SC.F32Fmaxf, p, B.call(SC.F32Fabs, B.call(SC.Bf16ToF32, x[e])))
        partials.append(p)
    warps = [F8._tree(B, partials[w * F8.WARP:(w + 1) * F8.WARP], SC.F32Fmaxf) for w in range(F8.QUANT_THREADS // F8.WARP)]
    m = warps[0]
    for w in warps[1:]:
        m = B.call(SC.F32Fmaxf, m, w)
    return m


@composite("Fp8RowScale", 2, [], lambda S: ((("absmax", F32),), F32), conformance="exact-model-tested")
def Fp8RowScaleV2(B, S, absmax):
    """`Fp8RowScale_v1` on bits: `fmaxf(absmax / 448.0f, 1.0f / (448.0f * 512.0f))`."""
    c = _Bits(B)
    return B.call(SC.F32Fmaxf, B.call(SC.F32Div, absmax, c.word(F448, 32)), c.word(F8.MIN_SCALE_BITS, 32))


def _quantize(B, c, xe, s, per_group: bool):
    """`cvt.rn.satfinite.e4m3(clamp(f32(x_e) / s, -448, 448))`, the clamp `fmaxf(-448, fminf(., 448))` (per-token) or
    `fminf(fmaxf(., -448), 448)` (per-group)."""
    v = B.call(SC.F32Div, B.call(SC.Bf16ToF32, xe), s)
    if per_group:
        v = B.call(SC.F32Fminf, B.call(SC.F32Fmaxf, v, c.word(F_NEG448, 32)), c.word(F448, 32))
    else:
        v = B.call(SC.F32Fmaxf, c.word(F_NEG448, 32), B.call(SC.F32Fminf, v, c.word(F448, 32)))
    return B.call(SC.F32ToE4m3Sat, v)


@composite("Fp8ActQuant", 2, ["K"], lambda S: ((("x", Array(S.K, BF16)),), Tuple(Array(S.K, E4M3), F32)),
           conformance="exact-model-tested")
def Fp8ActQuantV2(B, S, x):
    """`Fp8ActQuant_v1`'s body on bits: (x_q[K] e4m3, s_x f32) of one bf16 row."""
    c = _Bits(B)
    s = B.call(Fp8RowScaleV2, B.call(bind(Fp8RowAbsMaxV2, K=S.K), x))
    q = [_quantize(B, c, x[e], s, per_group=False) for e in range(S.K)]
    return tuple_of(Coll(Array(S.K, E4M3), concat([w.refs for w in q])), s)


@composite("DotE4m3", 2, ["K"], lambda S: ((("xq", Array(S.K, E4M3)), ("wq", Array(S.K, E4M3))), F32),
           conformance="exact-model-tested")
def DotE4m3V2(B, S, xq, wq):
    """`DotE4m3_v1`'s body on bits: K/32 `HopperE4m3QgmmaDot32_v2` steps in ascending k from +0."""
    assert S.K % 32 == 0, S.K
    acc = _Bits(B).word(0, 32)
    for s in range(S.K // 32):
        acc = B.call(BG.HopperE4m3QgmmaDot32, acc, xq[32 * s:32 * s + 32], wq[32 * s:32 * s + 32])
    return acc


@composite("ScaledMmFp8Coordinate", 2, ["K", "BIAS"],
           lambda S: (((("xq", Array(S.K, E4M3)), ("sx", F32), ("wrow", Array(S.K, E4M3)), ("sw", F32))
                       + ((("b", BF16),) if S.BIAS else ())), BF16),
           conformance="exact-model-tested")
def ScaledMmFp8CoordinateV2(B, S, xq, sx, wrow, sw, *bias):
    """`ScaledMmFp8Coordinate_v1`'s body on bits: `bf16(fma(s_w, s_x * acc, f32(b)))` or `bf16(s_w * (s_x * acc))`."""
    acc = B.call(bind(DotE4m3V2, K=S.K), xq, wrow)
    t = B.call(SC.F32Mul, sx, acc)
    if S.BIAS:
        y = B.call(SC.F32Fma, sw, t, B.call(SC.Bf16ToF32, bias[0]))
    else:
        y = B.call(SC.F32Mul, sw, t)
    return B.call(BG.F2fpBf16, y)


@composite("ScaledMmFp8", 2, ["K", "N", "BIAS"],
           lambda S: (((("xq", Array(S.K, E4M3)), ("sx", F32), ("w", Array(S.N, Array(S.K, E4M3))), ("sw", F32))
                       + ((("b", Array(S.N, BF16)),) if S.BIAS else ())),
                      Array(S.N, BF16)),
           conformance="exact-model-tested")
def ScaledMmFp8V2(B, S, xq, sx, w, sw, *bias):
    """`ScaledMmFp8_v1`'s body on bits: the N coordinates over the weight rows."""
    coord = bind(ScaledMmFp8CoordinateV2, K=S.K, BIAS=S.BIAS)
    if S.BIAS:
        return B.batch(coord, xq, sx, w, sw, bias[0], axes=(None, None, 0, None, 0))
    return B.batch(coord, xq, sx, w, sw, axes=(None, None, 0, None))


@composite("QuantLinearFp8", 2, ["K", "N", "BIAS"],
           lambda S: (((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, E4M3))), ("sw", F32))
                       + ((("b", Array(S.N, BF16)),) if S.BIAS else ())),
                      Tuple(Array(S.K, E4M3), F32, Array(S.N, BF16))),
           conformance="exact-model-tested")
def QuantLinearFp8V2(B, S, x, w, sw, *bias):
    """`QuantLinearFp8_v1`'s body on bits: `Fp8ActQuant_v2`, then `ScaledMmFp8_v2`; returns (x_q, s_x, y)."""
    q = B.call(bind(Fp8ActQuantV2, K=S.K), x)
    xq, sx = q[0], q[1]
    y = B.call(bind(ScaledMmFp8V2, K=S.K, N=S.N, BIAS=S.BIAS), xq, sx, w, sw, *bias)
    return tuple_of(xq, sx, y)


@composite("Fp8GroupScale", 2, ["G"], lambda S: ((("x", Array(S.G, BF16)),), F32), conformance="exact-model-tested")
def Fp8GroupScaleV2(B, S, x):
    """`Fp8GroupScale_v1`'s body on bits: `fmaxf`-fold from eps of `|f32(x_e)|`, divided by 448."""
    c = _Bits(B)
    m = c.word(EPS_QUANT, 32)
    for e in range(S.G):
        m = B.call(SC.F32Fmaxf, m, B.call(SC.F32Fabs, B.call(SC.Bf16ToF32, x[e])))
    return B.call(SC.F32Div, m, c.word(F448, 32))


@composite("Fp8GroupQuant", 2, ["K", "G"], lambda S: ((("x", Array(S.K, BF16)),), Tuple(Array(S.K, E4M3), Array(S.K // S.G, F32))),
           conformance="exact-model-tested")
def Fp8GroupQuantV2(B, S, x):
    """`Fp8GroupQuant_v1`'s body on bits: per group its scale and its e4m3 words."""
    K, G = S.K, S.G
    assert K % G == 0, (K, G)
    c = _Bits(B)
    scale = bind(Fp8GroupScaleV2, G=G)
    qs, ss = [], []
    for g in range(K // G):
        s = B.call(scale, x[g * G:(g + 1) * G])
        ss.append(s)
        qs.extend(_quantize(B, c, x[e], s, per_group=True) for e in range(g * G, (g + 1) * G))
    return tuple_of(Coll(Array(K, E4M3), concat([w.refs for w in qs])), Coll(Array(K // G, F32), concat([w.refs for w in ss])))


def _block_acc(B, c, K: int, G: int, xq, wrow, scale_of):
    """`acc = FFMA(temp, scale_of(kb), acc)` from +0 over the K/G tiles, `temp` the tile's G/32 k32 steps from +0, then the cast."""
    assert K % G == 0 and G % 32 == 0, (K, G)
    acc = c.word(0, 32)
    for kb in range(K // G):
        temp = c.word(0, 32)
        for st in range(G // 32):
            lo = kb * G + 32 * st
            temp = B.call(BG.HopperE4m3QgmmaDot32, temp, xq[lo:lo + 32], wrow[lo:lo + 32])
        acc = B.call(SC.F32Fma, temp, scale_of(kb), acc)
    return B.call(BG.F2fpBf16, acc)


@composite("ScaledMmFp8BlockCoordinate", 2, ["K", "G"],
           lambda S: ((("xq", Array(S.K, E4M3)), ("sx", Array(S.K // S.G, F32)), ("wrow", Array(S.K, E4M3)),
                       ("sw", Array(S.K // S.G, F32))), BF16),
           conformance="exact-model-tested")
def ScaledMmFp8BlockCoordinateV2(B, S, xq, sx, wrow, sw):
    """`ScaledMmFp8BlockCoordinate_v1`'s body on bits: the tile scale `FMUL(sx[kb], sw[kb])` inside the coordinate."""
    return _block_acc(B, _Bits(B), S.K, S.G, xq, wrow, lambda kb: B.call(SC.F32Mul, sx[kb], sw[kb]))


@composite("ScaledMmFp8Block", 2, ["K", "N", "G"],
           lambda S: ((("xq", Array(S.K, E4M3)), ("sx", Array(S.K // S.G, F32)), ("w", Array(S.N, Array(S.K, E4M3))),
                       ("ws", Array(S.N // S.G, Array(S.K // S.G, F32)))), Array(S.N, BF16)),
           conformance="exact-model-tested")
def ScaledMmFp8BlockV2(B, S, xq, sx, w, ws):
    """`ScaledMmFp8Block_v1`'s body on bits: column n over the weight row `w[n]` and the scale row `ws[n // G]`."""
    K, N, G = S.K, S.N, S.G
    assert N % G == 0, (N, G)
    coord = bind(ScaledMmFp8BlockCoordinateV2, K=K, G=G)
    parts = [B.batch(coord, xq, sx, w[nb * G:(nb + 1) * G], ws[nb], axes=(None, None, 0, None)) for nb in range(N // G)]
    return Coll(Array(N, BF16), concat([p.refs for p in parts]))


@composite("ScaledMmFp8BlockCoordinateGivenScale", 2, ["K", "G"],
           lambda S: ((("xq", Array(S.K, E4M3)), ("s", Array(S.K // S.G, F32)), ("wrow", Array(S.K, E4M3))), BF16),
           conformance="exact-model-tested")
def ScaledMmFp8BlockCoordinateGivenScaleV2(B, S, xq, s, wrow):
    """`ScaledMmFp8BlockCoordinateGivenScale_v1`'s body on bits: the tile scales read."""
    return _block_acc(B, _Bits(B), S.K, S.G, xq, wrow, lambda kb: s[kb])


@composite("ScaledMmFp8BlockSharedScaleTile", 2, ["K", "G"],
           lambda S: ((("xq", Array(S.K, E4M3)), ("sx", Array(S.K // S.G, F32)), ("w", Array(S.G, Array(S.K, E4M3))),
                       ("sw", Array(S.K // S.G, F32))), Array(S.G, BF16)),
           conformance="exact-model-tested")
def ScaledMmFp8BlockSharedScaleTileV2(B, S, xq, sx, w, sw):
    """`ScaledMmFp8BlockSharedScaleTile_v1`'s body on bits: the K/G scale products once, then the G coordinates reading them."""
    s = B.batch(SC.F32Mul, sx, sw, axes=(0, 0))
    return B.batch(bind(ScaledMmFp8BlockCoordinateGivenScaleV2, K=S.K, G=S.G), xq, s, w, axes=(None, None, 0))


@composite("ScaledMmFp8BlockSharedScale", 2, ["K", "N", "G"],
           lambda S: ((("xq", Array(S.K, E4M3)), ("sx", Array(S.K // S.G, F32)), ("w", Array(S.N, Array(S.K, E4M3))),
                       ("ws", Array(S.N // S.G, Array(S.K // S.G, F32)))), Array(S.N, BF16)),
           conformance="exact-model-tested")
def ScaledMmFp8BlockSharedScaleV2(B, S, xq, sx, w, ws):
    """`ScaledMmFp8BlockSharedScale_v1`'s body on bits: the N/G weight blocks (`ScaledMmFp8BlockSharedScaleTile_v2`)."""
    K, N, G = S.K, S.N, S.G
    assert N % G == 0, (N, G)
    blocks = w.reshape(Array(N // G, Array(G, Array(K, E4M3))))
    out = B.batch(bind(ScaledMmFp8BlockSharedScaleTileV2, K=K, G=G), xq, sx, blocks, ws, axes=(None, None, 0, 0))
    return Coll(Array(N, BF16), out.refs)


#: each Definition on bits by its word Definition
WORDS = {Fp8RowAbsMaxV2: F8.Fp8RowAbsMax, Fp8RowScaleV2: F8.Fp8RowScale, Fp8ActQuantV2: F8.Fp8ActQuant, DotE4m3V2: F8.DotE4m3,
         ScaledMmFp8CoordinateV2: F8.ScaledMmFp8Coordinate, ScaledMmFp8V2: F8.ScaledMmFp8, QuantLinearFp8V2: F8.QuantLinearFp8,
         Fp8GroupScaleV2: F8.Fp8GroupScale, Fp8GroupQuantV2: F8.Fp8GroupQuant,
         ScaledMmFp8BlockCoordinateV2: F8.ScaledMmFp8BlockCoordinate, ScaledMmFp8BlockV2: F8.ScaledMmFp8Block,
         ScaledMmFp8BlockCoordinateGivenScaleV2: F8.ScaledMmFp8BlockCoordinateGivenScale,
         ScaledMmFp8BlockSharedScaleTileV2: F8.ScaledMmFp8BlockSharedScaleTile,
         ScaledMmFp8BlockSharedScaleV2: F8.ScaledMmFp8BlockSharedScale}
for _bits, _word in WORDS.items():
    _bits.word = _view(_word)  # type: ignore[attr-defined]
