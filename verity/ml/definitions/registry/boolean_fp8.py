"""The served FP8 linear on bits: `fp8`'s composites, version 2 (the two dots versions 3 and 4, versions 1 and 2 being the
word dots), their word bodies line for line over Boolean Definitions, so each one's word view is its word composite at the same
statics (`elementwise.on_bits`; a `DOT` static, a Boolean e4m3 step, read as its word view).

    Fp8RowAbsMax_v2{K}, Fp8RowScale_v2, Fp8ActQuant_v2{K}           the per-token activation quantisation
    DotE4m3_v3{K}, ScaledMmFp8Coordinate_v2{K,BIAS}, ScaledMmFp8_v2{K,N,BIAS}, QuantLinearFp8_v2{K,N,BIAS}
                                                                    the per-tensor cutlass GEMM (k32 wgmma chain, fused epilogue)
    DotE4m3_v4{K,DOT}, ScaledMmFp8NoSwapCoordinate_v2{K,BIAS,DOT}, ScaledMmFp8NoSwap_v2{K,N,BIAS,DOT}
                                                                    the same GEMM without swap_ab over the step DOT (cc 12.0)
    Fp8GroupScale_v2{G}, Fp8GroupQuant_v2{K,G}, ScaledMmFp8BlockCoordinate_v2{K,G}, ScaledMmFp8Block_v2{K,N,G},
    ScaledMmFp8BlockCoordinateGivenScale_v2{K,G}, ScaledMmFp8BlockSharedScaleTile_v2{K,G}, ScaledMmFp8BlockSharedScale_v2{K,N,G}
                                                                    the block-scaled linear (128x128 weight blocks)
    ScaledMmFp8BlockCoordinate_v4{K,G,DOT}, ScaledMmFp8Block_v4{K,N,G,DOT}, ScaledMmFp8BlockCoordinateGivenScale_v4{K,G,DOT},
    ScaledMmFp8BlockSharedScaleTile_v4{K,G,DOT}, ScaledMmFp8BlockSharedScale_v4{K,N,G,DOT}
                                                                    the same over the step DOT (the word versions 3)

Every word is its bits (`verity.ml.circuits.boolean.word_type`): bf16 `Array<16, Value<1>>`, f32 `Array<32, Value<1>>`, e4m3
`Array<8, Value<1>>`. The callees are the Boolean versions of the word bodies': `verity.ml.circuits.ops.boolean.scalar`'s f32 arithmetic,
max / min and casts, `verity.ml.circuits.ops.boolean.gemm`'s `HopperE4m3QgmmaDot32_v2` (or `DOT`, such as `BlackwellE4m3QmmaDot32_v2`) and
`F2fpBf16_v2`; the word constants (+0, 448, -448, the minimum scale, the group quantiser's eps) are written with the constant
bits.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import scalar as SC
from verity.ml.definitions.registry import fp8 as F8
from verity.ml.definitions.registry.boolean_attention import _Bits
from verity.ml.definitions.registry.prims import bits_of

BF16 = Array(16, BL.BIT)
F32 = Array(32, BL.BIT)
E4M3 = Array(8, BL.BIT)

F448 = bits_of(448.0)
F_NEG448 = bits_of(-448.0)
EPS_QUANT = bits_of(1e-10)


def _row_absmax(B, S, x):
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


Fp8RowAbsMaxV2 = EW.on_bits("Fp8RowAbsMax", 2, ["K"], lambda S: ((("x", Array(S.K, BF16)),), F32), _row_absmax, F8.Fp8RowAbsMax)


def _row_scale(B, S, absmax):
    """`Fp8RowScale_v1` on bits: `fmaxf(absmax / 448.0f, 1.0f / (448.0f * 512.0f))`."""
    c = _Bits(B)
    return B.call(SC.F32Fmaxf, B.call(SC.F32Div, absmax, c.word(F448, 32)), c.word(F8.MIN_SCALE_BITS, 32))


Fp8RowScaleV2 = EW.on_bits("Fp8RowScale", 2, [], lambda S: ((("absmax", F32),), F32), _row_scale, F8.Fp8RowScale)


def _quantize(B, c, xe, s, per_group: bool):
    """`cvt.rn.satfinite.e4m3(clamp(f32(x_e) / s, -448, 448))`, the clamp `fmaxf(-448, fminf(., 448))` (per-token) or
    `fminf(fmaxf(., -448), 448)` (per-group)."""
    v = B.call(SC.F32Div, B.call(SC.Bf16ToF32, xe), s)
    if per_group:
        v = B.call(SC.F32Fminf, B.call(SC.F32Fmaxf, v, c.word(F_NEG448, 32)), c.word(F448, 32))
    else:
        v = B.call(SC.F32Fmaxf, c.word(F_NEG448, 32), B.call(SC.F32Fminf, v, c.word(F448, 32)))
    return B.call(SC.F32ToE4m3Sat, v)


def _act_quant(B, S, x):
    """`Fp8ActQuant_v1`'s body on bits: (x_q[K] e4m3, s_x f32) of one bf16 row."""
    c = _Bits(B)
    s = B.call(Fp8RowScaleV2, B.call(bind(Fp8RowAbsMaxV2, K=S.K), x))
    q = [_quantize(B, c, x[e], s, per_group=False) for e in range(S.K)]
    return tuple_of(Coll(Array(S.K, E4M3), concat([w.refs for w in q])), s)


Fp8ActQuantV2 = EW.on_bits("Fp8ActQuant", 2, ["K"], lambda S: ((("x", Array(S.K, BF16)),), Tuple(Array(S.K, E4M3), F32)),
                         _act_quant, F8.Fp8ActQuant)


def _dot_e4m3(B, S, xq, wq):
    """`DotE4m3_v1`'s body on bits: K/32 `HopperE4m3QgmmaDot32_v2` steps in ascending k from +0."""
    assert S.K % 32 == 0, S.K
    acc = _Bits(B).word(0, 32)
    for s in range(S.K // 32):
        acc = B.call(BG.HopperE4m3QgmmaDot32, acc, xq[32 * s:32 * s + 32], wq[32 * s:32 * s + 32])
    return acc


DotE4m3V3 = EW.on_bits("DotE4m3", 3, ["K"], lambda S: ((("xq", Array(S.K, E4M3)), ("wq", Array(S.K, E4M3))), F32), _dot_e4m3,
                     F8.DotE4m3)


def _coordinate(B, S, xq, sx, wrow, sw, *bias):
    """`ScaledMmFp8Coordinate_v1`'s body on bits: `bf16(fma(s_w, s_x * acc, f32(b)))` or `bf16(s_w * (s_x * acc))`."""
    acc = B.call(bind(DotE4m3V3, K=S.K), xq, wrow)
    t = B.call(SC.F32Mul, sx, acc)
    if S.BIAS:
        y = B.call(SC.F32Fma, sw, t, B.call(SC.Bf16ToF32, bias[0]))
    else:
        y = B.call(SC.F32Mul, sw, t)
    return B.call(BG.F2fpBf16, y)


ScaledMmFp8CoordinateV2 = EW.on_bits(
    "ScaledMmFp8Coordinate", 2, ["K", "BIAS"],
    lambda S: (((("xq", Array(S.K, E4M3)), ("sx", F32), ("wrow", Array(S.K, E4M3)), ("sw", F32))
                + ((("b", BF16),) if S.BIAS else ())), BF16),
    _coordinate, F8.ScaledMmFp8Coordinate)


def _scaled_mm(B, S, xq, sx, w, sw, *bias):
    """`ScaledMmFp8_v1`'s body on bits: the N coordinates over the weight rows."""
    coord = bind(ScaledMmFp8CoordinateV2, K=S.K, BIAS=S.BIAS)
    if S.BIAS:
        return B.batch(coord, xq, sx, w, sw, bias[0], axes=(None, None, 0, None, 0))
    return B.batch(coord, xq, sx, w, sw, axes=(None, None, 0, None))


ScaledMmFp8V2 = EW.on_bits(
    "ScaledMmFp8", 2, ["K", "N", "BIAS"],
    lambda S: (((("xq", Array(S.K, E4M3)), ("sx", F32), ("w", Array(S.N, Array(S.K, E4M3))), ("sw", F32))
                + ((("b", Array(S.N, BF16)),) if S.BIAS else ())),
               Array(S.N, BF16)),
    _scaled_mm, F8.ScaledMmFp8)


def _quant_linear(B, S, x, w, sw, *bias):
    """`QuantLinearFp8_v1`'s body on bits: `Fp8ActQuant_v2`, then `ScaledMmFp8_v2`; returns (x_q, s_x, y)."""
    q = B.call(bind(Fp8ActQuantV2, K=S.K), x)
    xq, sx = q[0], q[1]
    y = B.call(bind(ScaledMmFp8V2, K=S.K, N=S.N, BIAS=S.BIAS), xq, sx, w, sw, *bias)
    return tuple_of(xq, sx, y)


QuantLinearFp8V2 = EW.on_bits(
    "QuantLinearFp8", 2, ["K", "N", "BIAS"],
    lambda S: (((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, E4M3))), ("sw", F32))
                + ((("b", Array(S.N, BF16)),) if S.BIAS else ())),
               Tuple(Array(S.K, E4M3), F32, Array(S.N, BF16))),
    _quant_linear, F8.QuantLinearFp8)


def _dot_e4m3_steps(B, S, xq, wq):
    """`DotE4m3_v2`'s body on bits: K/32 `DOT` steps (a Boolean e4m3 k32 step) in ascending k from +0."""
    assert S.K % 32 == 0, S.K
    acc = _Bits(B).word(0, 32)
    for s in range(S.K // 32):
        acc = B.call(S.DOT, acc, xq[32 * s:32 * s + 32], wq[32 * s:32 * s + 32])
    return acc


DotE4m3V4 = EW.on_bits("DotE4m3", 4, ["K", "DOT"], lambda S: ((("xq", Array(S.K, E4M3)), ("wq", Array(S.K, E4M3))), F32),
                       _dot_e4m3_steps, F8.DotE4m3V2, lifted=["DOT"])


def _no_swap_coordinate(B, S, xq, sx, wrow, sw, *bias):
    """`ScaledMmFp8NoSwapCoordinate_v1`'s body on bits: `bf16(s_x * (s_w * acc))` or `bf16(fma(s_x, s_w * acc, f32(b)))`, `acc`
    the `DotE4m3_v4{K, DOT}` chain."""
    acc = B.call(bind(DotE4m3V4, K=S.K, DOT=S.DOT), xq, wrow)
    t = B.call(SC.F32Mul, sw, acc)
    if S.BIAS:
        y = B.call(SC.F32Fma, sx, t, B.call(SC.Bf16ToF32, bias[0]))
    else:
        y = B.call(SC.F32Mul, sx, t)
    return B.call(BG.F2fpBf16, y)


ScaledMmFp8NoSwapCoordinateV2 = EW.on_bits(
    "ScaledMmFp8NoSwapCoordinate", 2, ["K", "BIAS", "DOT"],
    lambda S: (((("xq", Array(S.K, E4M3)), ("sx", F32), ("wrow", Array(S.K, E4M3)), ("sw", F32))
                + ((("b", BF16),) if S.BIAS else ())), BF16),
    _no_swap_coordinate, F8.ScaledMmFp8NoSwapCoordinate, lifted=["DOT"])


def _no_swap(B, S, xq, sx, w, sw, *bias):
    """`ScaledMmFp8NoSwap_v1`'s body on bits: the N coordinates (`ScaledMmFp8NoSwapCoordinate_v2`) over the weight rows."""
    coord = bind(ScaledMmFp8NoSwapCoordinateV2, K=S.K, BIAS=S.BIAS, DOT=S.DOT)
    if S.BIAS:
        return B.batch(coord, xq, sx, w, sw, bias[0], axes=(None, None, 0, None, 0))
    return B.batch(coord, xq, sx, w, sw, axes=(None, None, 0, None))


ScaledMmFp8NoSwapV2 = EW.on_bits(
    "ScaledMmFp8NoSwap", 2, ["K", "N", "BIAS", "DOT"],
    lambda S: (((("xq", Array(S.K, E4M3)), ("sx", F32), ("w", Array(S.N, Array(S.K, E4M3))), ("sw", F32))
                + ((("b", Array(S.N, BF16)),) if S.BIAS else ())),
               Array(S.N, BF16)),
    _no_swap, F8.ScaledMmFp8NoSwap, lifted=["DOT"])


def _group_scale(B, S, x):
    """`Fp8GroupScale_v1`'s body on bits: `fmaxf`-fold from eps of `|f32(x_e)|`, divided by 448."""
    c = _Bits(B)
    m = c.word(EPS_QUANT, 32)
    for e in range(S.G):
        m = B.call(SC.F32Fmaxf, m, B.call(SC.F32Fabs, B.call(SC.Bf16ToF32, x[e])))
    return B.call(SC.F32Div, m, c.word(F448, 32))


Fp8GroupScaleV2 = EW.on_bits("Fp8GroupScale", 2, ["G"], lambda S: ((("x", Array(S.G, BF16)),), F32), _group_scale, F8.Fp8GroupScale)


def _group_quant(B, S, x):
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


Fp8GroupQuantV2 = EW.on_bits(
    "Fp8GroupQuant", 2, ["K", "G"], lambda S: ((("x", Array(S.K, BF16)),), Tuple(Array(S.K, E4M3), Array(S.K // S.G, F32))),
    _group_quant, F8.Fp8GroupQuant)


def _block_acc(B, c, K: int, G: int, xq, wrow, scale_of, dot=BG.HopperE4m3QgmmaDot32):
    """`acc = FFMA(temp, scale_of(kb), acc)` from +0 over the K/G tiles, `temp` the tile's G/32 k32 steps `dot` from +0, then the cast."""
    assert K % G == 0 and G % 32 == 0, (K, G)
    acc = c.word(0, 32)
    for kb in range(K // G):
        temp = c.word(0, 32)
        for st in range(G // 32):
            lo = kb * G + 32 * st
            temp = B.call(dot, temp, xq[lo:lo + 32], wrow[lo:lo + 32])
        acc = B.call(SC.F32Fma, temp, scale_of(kb), acc)
    return B.call(BG.F2fpBf16, acc)


def _coordinate_sig(S):
    return (("xq", Array(S.K, E4M3)), ("sx", Array(S.K // S.G, F32)), ("wrow", Array(S.K, E4M3)), ("sw", Array(S.K // S.G, F32))), BF16


def _block_sig(S):
    return ((("xq", Array(S.K, E4M3)), ("sx", Array(S.K // S.G, F32)), ("w", Array(S.N, Array(S.K, E4M3))),
             ("ws", Array(F8.weight_blocks(S.N, S.G), Array(S.K // S.G, F32)))), Array(S.N, BF16))


def _given_scale_sig(S):
    return (("xq", Array(S.K, E4M3)), ("s", Array(S.K // S.G, F32)), ("wrow", Array(S.K, E4M3))), BF16


def _tile_sig(S):
    return ((("xq", Array(S.K, E4M3)), ("sx", Array(S.K // S.G, F32)), ("w", Array(S.G, Array(S.K, E4M3))), ("sw", Array(S.K // S.G, F32))),
            Array(S.G, BF16))


def _block_columns(B, K: int, N: int, G: int, coord, xq, sx, w, ws):
    parts = [B.batch(coord, xq, sx, w[nb * G:min((nb + 1) * G, N)], ws[nb], axes=(None, None, 0, None))
             for nb in range(F8.weight_blocks(N, G))]
    return Coll(Array(N, BF16), concat([p.refs for p in parts]))


def _shared_scale_blocks(B, K: int, N: int, G: int, tile, given, xq, sx, w, ws):
    full, r = divmod(N, G)
    if not r:
        blocks = w.reshape(Array(N // G, Array(G, Array(K, E4M3))))
        out = B.batch(tile, xq, sx, blocks, ws, axes=(None, None, 0, 0))
        return Coll(Array(N, BF16), out.refs)
    parts = []
    if full:
        blocks = Coll(Array(full, Array(G, Array(K, E4M3))), w[0:full * G].refs)
        scales = Coll(Array(full, Array(K // G, F32)), ws[0:full].refs)
        parts.append(B.batch(tile, xq, sx, blocks, scales, axes=(None, None, 0, 0)))
    s = B.batch(SC.F32Mul, sx, ws[full], axes=(0, 0))
    parts.append(B.batch(given, xq, s, w[full * G:N], axes=(None, None, 0)))
    return Coll(Array(N, BF16), concat([p.refs for p in parts]))


def _block_coordinate(B, S, xq, sx, wrow, sw):
    """`ScaledMmFp8BlockCoordinate_v1`'s body on bits: the tile scale `FMUL(sx[kb], sw[kb])` inside the coordinate."""
    return _block_acc(B, _Bits(B), S.K, S.G, xq, wrow, lambda kb: B.call(SC.F32Mul, sx[kb], sw[kb]))


ScaledMmFp8BlockCoordinateV2 = EW.on_bits("ScaledMmFp8BlockCoordinate", 2, ["K", "G"], _coordinate_sig, _block_coordinate,
                                          F8.ScaledMmFp8BlockCoordinate)


def _scaled_mm_block(B, S, xq, sx, w, ws):
    """`ScaledMmFp8Block_v1`'s body on bits: column n over the weight row `w[n]` and the scale row `ws[n // G]` (a partial last
    weight block when G does not divide N)."""
    K, N, G = S.K, S.N, S.G
    return _block_columns(B, K, N, G, bind(ScaledMmFp8BlockCoordinateV2, K=K, G=G), xq, sx, w, ws)


ScaledMmFp8BlockV2 = EW.on_bits("ScaledMmFp8Block", 2, ["K", "N", "G"], _block_sig, _scaled_mm_block, F8.ScaledMmFp8Block)


def _block_coordinate_given_scale(B, S, xq, s, wrow):
    """`ScaledMmFp8BlockCoordinateGivenScale_v1`'s body on bits: the tile scales read."""
    return _block_acc(B, _Bits(B), S.K, S.G, xq, wrow, lambda kb: s[kb])


ScaledMmFp8BlockCoordinateGivenScaleV2 = EW.on_bits("ScaledMmFp8BlockCoordinateGivenScale", 2, ["K", "G"], _given_scale_sig,
                                                    _block_coordinate_given_scale, F8.ScaledMmFp8BlockCoordinateGivenScale)


def _shared_scale_tile(B, S, xq, sx, w, sw):
    """`ScaledMmFp8BlockSharedScaleTile_v1`'s body on bits: the K/G scale products once, then the G coordinates reading them."""
    s = B.batch(SC.F32Mul, sx, sw, axes=(0, 0))
    return B.batch(bind(ScaledMmFp8BlockCoordinateGivenScaleV2, K=S.K, G=S.G), xq, s, w, axes=(None, None, 0))


ScaledMmFp8BlockSharedScaleTileV2 = EW.on_bits("ScaledMmFp8BlockSharedScaleTile", 2, ["K", "G"], _tile_sig, _shared_scale_tile,
                                               F8.ScaledMmFp8BlockSharedScaleTile)


def _scaled_mm_block_shared_scale(B, S, xq, sx, w, ws):
    """`ScaledMmFp8BlockSharedScale_v1`'s body on bits: the N/G whole weight blocks (`ScaledMmFp8BlockSharedScaleTile_v2`), then a
    partial last block's products and coordinates in this body."""
    K, N, G = S.K, S.N, S.G
    return _shared_scale_blocks(B, K, N, G, bind(ScaledMmFp8BlockSharedScaleTileV2, K=K, G=G), bind(ScaledMmFp8BlockCoordinateGivenScaleV2, K=K, G=G),
                                xq, sx, w, ws)


ScaledMmFp8BlockSharedScaleV2 = EW.on_bits("ScaledMmFp8BlockSharedScale", 2, ["K", "N", "G"], _block_sig, _scaled_mm_block_shared_scale,
                                           F8.ScaledMmFp8BlockSharedScale)


def _block_coordinate_dot(B, S, xq, sx, wrow, sw):
    """`ScaledMmFp8BlockCoordinate_v3`'s body on bits: the tiles' steps `DOT` (a Boolean e4m3 k32 step), the tile scale inside."""
    return _block_acc(B, _Bits(B), S.K, S.G, xq, wrow, lambda kb: B.call(SC.F32Mul, sx[kb], sw[kb]), S.DOT)


ScaledMmFp8BlockCoordinateV4 = EW.on_bits("ScaledMmFp8BlockCoordinate", 4, ["K", "G", "DOT"], _coordinate_sig, _block_coordinate_dot,
                                          F8.ScaledMmFp8BlockCoordinateV3, lifted=["DOT"])


def _scaled_mm_block_dot(B, S, xq, sx, w, ws):
    """`ScaledMmFp8Block_v3`'s body on bits: the N coordinates (`ScaledMmFp8BlockCoordinate_v4`) over the step `DOT`."""
    K, N, G = S.K, S.N, S.G
    return _block_columns(B, K, N, G, bind(ScaledMmFp8BlockCoordinateV4, K=K, G=G, DOT=S.DOT), xq, sx, w, ws)


ScaledMmFp8BlockV4 = EW.on_bits("ScaledMmFp8Block", 4, ["K", "N", "G", "DOT"], _block_sig, _scaled_mm_block_dot, F8.ScaledMmFp8BlockV3,
                                lifted=["DOT"])


def _block_coordinate_given_scale_dot(B, S, xq, s, wrow):
    """`ScaledMmFp8BlockCoordinateGivenScale_v3`'s body on bits: the tile scales read, the steps `DOT`."""
    return _block_acc(B, _Bits(B), S.K, S.G, xq, wrow, lambda kb: s[kb], S.DOT)


ScaledMmFp8BlockCoordinateGivenScaleV4 = EW.on_bits("ScaledMmFp8BlockCoordinateGivenScale", 4, ["K", "G", "DOT"], _given_scale_sig,
                                                    _block_coordinate_given_scale_dot, F8.ScaledMmFp8BlockCoordinateGivenScaleV3,
                                                    lifted=["DOT"])


def _shared_scale_tile_dot(B, S, xq, sx, w, sw):
    """`ScaledMmFp8BlockSharedScaleTile_v3`'s body on bits: the K/G scale products once, then the G coordinates over the step `DOT`."""
    s = B.batch(SC.F32Mul, sx, sw, axes=(0, 0))
    return B.batch(bind(ScaledMmFp8BlockCoordinateGivenScaleV4, K=S.K, G=S.G, DOT=S.DOT), xq, s, w, axes=(None, None, 0))


ScaledMmFp8BlockSharedScaleTileV4 = EW.on_bits("ScaledMmFp8BlockSharedScaleTile", 4, ["K", "G", "DOT"], _tile_sig, _shared_scale_tile_dot,
                                               F8.ScaledMmFp8BlockSharedScaleTileV3, lifted=["DOT"])


def _scaled_mm_block_shared_scale_dot(B, S, xq, sx, w, ws):
    """`ScaledMmFp8BlockSharedScale_v3`'s body on bits: the whole weight blocks (`ScaledMmFp8BlockSharedScaleTile_v4`), then a partial
    last block's products and coordinates in this body."""
    K, N, G, DOT = S.K, S.N, S.G, S.DOT
    return _shared_scale_blocks(B, K, N, G, bind(ScaledMmFp8BlockSharedScaleTileV4, K=K, G=G, DOT=DOT),
                                bind(ScaledMmFp8BlockCoordinateGivenScaleV4, K=K, G=G, DOT=DOT), xq, sx, w, ws)


ScaledMmFp8BlockSharedScaleV4 = EW.on_bits("ScaledMmFp8BlockSharedScale", 4, ["K", "N", "G", "DOT"], _block_sig,
                                           _scaled_mm_block_shared_scale_dot, F8.ScaledMmFp8BlockSharedScaleV3, lifted=["DOT"])
