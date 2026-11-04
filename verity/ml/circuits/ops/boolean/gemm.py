"""The BF16 GEMM coordinate as Boolean Definitions: the next version of each word Definition, the same function on bits.

    F2fpBf16_v2                 (a: Array<32, Value<1>>) -> Array<16, Value<1>>                       word view: F2fpBf16_v1
    AmpereBF16TcDot16_v3        (acc: Array<32, Value<1>>, a, b: Array<16, Array<16, Value<1>>>)
                                -> Array<32, Value<1>>                                                word view: AmpereBF16TcDot16_v2
    HopperBF16WgmmaDot16_v2     the same signature                                                    word view: HopperBF16WgmmaDot16_v1
    HopperBF16WgmmaDot16_v3     the same signature                                                    word view: HopperBF16WgmmaDot16_v1
    DotBf16_v3{K, DOT}          (x, w: Array<K, Array<16, Value<1>>>) -> Array<32, Value<1>>          word view: DotBf16_v2{K, DOT'}
    GemmCoordinate_v3{K, DOT}   (x, wrow: Array<K, Array<16, Value<1>>>) -> Array<16, Value<1>>       word view: GemmCoordinate_v2{K, DOT'}
    GemmCoordinatePad_v2{T, K, DOT}  (x, wrow: Array<T, Array<16, Value<1>>>) -> Array<16, Value<1>>  word view: GemmCoordinatePad_v1{T, K, DOT'}
    PvRow_v2{T, K, D, DOT}      (p: Array<T, bf16 bits>, v0..v{T-1}: Array<D, bf16 bits>) -> Array<D, Array<16, Value<1>>>
                                                                                                      word view: PvRow_v1{T, K, D, DOT'}
    PvRowPad_v2{T, K, D, DOT}   (p, v0..v{T-1} as PvRow_v2's, n: Array<16, Value<1>>) -> as PvRow_v2's
                                                                                                      word view: PvRowPad_v1{T, K, D, DOT'}
    Gemm_v3{K, N, DOT}          (x: Array<K, bf16 bits>, w: Array<N, Array<K, bf16 bits>>)
                                -> Array<N, Array<16, Value<1>>>                                      word view: Gemm_v2{K, N, DOT'}
    HopperE4m3QgmmaDot32_v2     (acc: Array<32, Value<1>>, a, b: Array<32, Array<8, Value<1>>>)
                                -> Array<32, Value<1>>                                                word view: HopperE4m3QgmmaDot32_v1
    HopperE4m3MmaV2Dot32_v2     the same signature                                                    word view: HopperE4m3MmaV2Dot32_v1
    BlackwellE4m3QmmaDot32_v2   the same signature                                                    word view: BlackwellE4m3QmmaDot32_v1
    BlackwellE5m2QmmaDot32_v2   the same signature (E5M2 operands)                                    word view: BlackwellE5m2QmmaDot32_v1
    GemmCoordinateE4m3_v2{K, DOT}  (x, wrow: Array<K, Array<8, Value<1>>>) -> Array<16, Value<1>>     word view: GemmCoordinateE4m3_v1{K, DOT'}
    BlackwellNvf4OmmaDot64_v2   (acc, a, b: Array<64, Array<4, Value<1>>>, sa, sb: Array<4, Array<8, Value<1>>>)
                                -> Array<32, Value<1>>                                                word view: BlackwellNvf4OmmaDot64_v1
    BlackwellMxf4OmmaDot64_v2   the same with sa, sb: Array<2, Array<8, Value<1>>>                    word view: BlackwellMxf4OmmaDot64_v1
    GemmCoordinateNvf4_v2{K, DOT}, GemmCoordinateMxf4_v2{K, DOT}  (x, wrow, sx, sw) -> bf16 bits      word views: their v1{K, DOT'}
    BlackwellS8ImmaDot32_v2     (acc: Array<32, Value<1>>, a, b: Array<32, Array<8, Value<1>>>)
                                -> Array<32, Value<1>>                                                word view: BlackwellS8ImmaDot32_v1
    S32ToF32Rn_v2               (a: Array<32, Value<1>>) -> Array<32, Value<1>>                       word view: S32ToF32Rn_v1
    GemmAccumulatorS8_v2{K, DOT}    (x, wrow: Array<K, Array<8, Value<1>>>) -> Array<32, Value<1>>   word view: GemmAccumulatorS8_v1{K, DOT'}
    ScaledMmS8Coordinate_v2{K, DOT} (x, sx, wrow, sw) -> Array<16, Value<1>>                          word view: ScaledMmS8Coordinate_v1{K, DOT'}
    ScaledMmS8CoordinateRows_v2{K, R, DOT}  (x0..x{R-1}, sx, wrow0..wrow{R-1}, sw) -> Array<16, Value<1>>
                                                                                                      word view: ScaledMmS8CoordinateRows_v1{K, R, DOT'}
    ScaledMmS8ResidualCoordinateRows_v2{K, R, DOT}  (ScaledMmS8CoordinateRows_v2's ports, r: Array<16, Value<1>>)
                                -> Array<16, Value<1>>                                    word view: ScaledMmS8ResidualCoordinateRows_v1{K, R, DOT'}
    ScaledMmS8_v2{K, N, DOT}    (x: Array<K, s8 bits>, sx, w: Array<N, Array<K, s8 bits>>, sw: Array<N, f32 bits>)
                                -> Array<N, Array<16, Value<1>>>                                      word view: ScaledMmS8_v1{K, N, DOT'}

The steps and the cast are traced from `fp`'s builders (`trace.definition`): their bodies are And/Xor/Not gates, with exactly
the builders' live ANDs. `HopperBF16WgmmaDot16_v3` is v2 without the finite stand-ins (`fp.tc_dot16`'s ``stand_ins=False``,
whose docstring proves it the same function); it is ``opt_in``: `boolean_version` never picks it, so it serves only the
callers that bind it (the training step's GEMM routes) and the served Programs keep v2. `GemmCoordinate_v3` is `GemmCoordinate_v2`'s body on bits: K/16 calls of `DOT` from the zero
accumulator (32 constant-0 bits), then `F2fpBf16_v2`; `DotBf16_v3` is the same steps without the cast, and `Gemm_v3` is
`Gemm_v2`'s: the coordinate batched over W's N rows. `GemmCoordinatePad_v2` is `GemmCoordinate_v3` on its T-word operands
followed by K - T words of constant-0 bits, and `PvRow_v2` is that coordinate batched over the D columns of V's T key rows;
`PvRowPad_v2` is `PvRow_v2` on its operands with key t's words replaced by constant-0 bits unless t < n, n unsigned
(`PvRowPad_v1`'s compare on bits: `Bf16ToF32_v2`, `I32Add_v2`, `I32Le_v2`; then `SelectBf16_v2`).
The E4M3 steps are `fp.tc_dot_e4m3` at each step's groups, adder width and floor
(`HopperE4m3MmaV2Dot32_v2` on the operands in `MMAV2_E4M3_K_ORDER`; the E5M2 step `fp.tc_dot_e5m2`),
and `GemmCoordinateE4m3_v2` is K/32 of them, then the cast; the FP4 steps are `fp4.step_total` and their coordinates K/64 of
them. The s8 step is `integer.step_s8_bits` (Baugh-Wooley arrays, one carry-save reduction modulo 2^33, the clamp at D), the
int-to-float cast `fp.s32_to_f32`, and `ScaledMmS8Coordinate_v2` the accumulator, the cast, two `F32Mul_v4` and `F2fpBf16_v2`;
`ScaledMmS8CoordinateRows_v2` is that coordinate on its operands' R parts concatenated, and
`ScaledMmS8ResidualCoordinateRows_v2` adds its word to r with `Bf16AddF2fp_v2`; `ScaledMmS8_v2` is that coordinate
batched over W's N rows and their scales; `GEMM_SM90_S8` binds them on the H200 with the
same step (`s8_gemm`, `G.s8_gemm`'s check on bits). Each names its word
Definition (`trace.word_of`; ``DOT'`` is `DOT`'s); the tests check that the two agree, on random and special words and on the
hardware captures. Each composite refuses at `bind` the K its word Definition refuses (its `refuse`). Each step is bound to its
word step's silicon model (`census.silicon_step`), so the census counts its gates as `tc:<precision>`, and `GemmAccumulatorS8_v2`
serves none of its products, as its word view (`census.serves_none`).
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.census import FAMILY_COUNTS, serves_none, silicon_step
from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Coll, broadcast, concat
from verity.ml.circuits.types import Array

from .. import gemm as G
from .. import prims as P
from . import elementwise as EW
from . import scalar as SC
from verity.ml.circuits.silicon import fp, fp4
from verity.ml.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32, HOPPER_SM90_S8_K32, step_s8_bits
from verity.ml.circuits.silicon.total_fp8 import MMAV2_E4M3_K_ORDER
from verity.ml.circuits.boolean.trace import definition, word_of

__all__ = ["AmpereBF16TcDot16", "BlackwellE4m3QmmaDot32", "BlackwellE5m2QmmaDot32", "BlackwellMxf4OmmaDot64", "BlackwellNvf4OmmaDot64",
           "BlackwellS8ImmaDot32", "DotBf16", "F2fpBf16",
           "GEMM_AMPERE", "GEMM_HOPPER", "GEMM_HOPPER_E4M3", "GEMM_HOPPER_E4M3_MMAV2", "GEMM_SM120_E4M3", "GEMM_SM120_MXF4", "GEMM_SM120_NVF4",
           "GEMM_SM120_S8", "GEMM_SM90_S8", "Gemm", "GemmAccumulatorS8",
           "GemmCoordinate", "GemmCoordinateE4m3", "GemmCoordinateMxf4", "GemmCoordinateNvf4", "GemmCoordinatePad",
           "HopperBF16WgmmaDot16", "HopperBF16WgmmaDot16V3", "HopperE4m3MmaV2Dot32", "HopperE4m3QgmmaDot32", "PvRow", "PvRowPad", "S32ToF32Rn", "ScaledMmS8",
           "ScaledMmS8Coordinate", "ScaledMmS8CoordinateRows", "ScaledMmS8ResidualCoordinateRows", "s8_gemm"]

BIT = BL.BIT
F32 = Array(32, BIT)
BF16 = Array(16, BIT)
A16 = Array(16, BF16)
E4M3 = Array(8, BIT)
A32_8 = Array(32, E4M3)
E2M1 = Array(4, BIT)
A64_4 = Array(64, E2M1)
SCALE = Array(8, BIT)
A4_8 = Array(4, SCALE)
A2_8 = Array(2, SCALE)

#: F2fpBf16_v1's NaN: every NaN input gives 0x7FFF
NAN_F2FP = 0x7FFF

F2fpBf16 = definition("F2fpBf16", 2, lambda C, a: fp.f32_to_bf16(C, a, NAN_F2FP), (("a", F32),), BF16, word=P.F2fpBf16,
                      doc="cvt.rn.bf16.f32 on bits: F2fpBf16_v1 (round to nearest even, NaN -> 0x7FFF)")
AmpereBF16TcDot16 = definition(
    "AmpereBF16TcDot16", 3, lambda C, acc, *ab: fp.tc_dot16(C, acc, list(ab[:16]), list(ab[16:])),
    (("acc", F32), ("a", A16), ("b", A16)), F32, word=P.AmpereBF16TcDot16,
    doc="mma.sync.m16n8k16 bf16, one accumulator coordinate, on bits: AmpereBF16TcDot16_v2")
HopperBF16WgmmaDot16 = definition(
    "HopperBF16WgmmaDot16", 2,
    lambda C, acc, *ab: fp.tc_dot16(C, acc, list(ab[:16]), list(ab[16:]), groups=(16,), W=26, F=-133),
    (("acc", F32), ("a", A16), ("b", A16)), F32, word=P.HopperBF16WgmmaDot16,
    doc="wgmma.m64nNk16 bf16, one accumulator coordinate, on bits: HopperBF16WgmmaDot16_v1")
HopperBF16WgmmaDot16V3 = definition(
    "HopperBF16WgmmaDot16", 3,
    lambda C, acc, *ab: fp.tc_dot16(C, acc, list(ab[:16]), list(ab[16:]), groups=(16,), W=26, F=-133, stand_ins=False),
    (("acc", F32), ("a", A16), ("b", A16)), F32, word=P.HopperBF16WgmmaDot16,
    doc="wgmma.m64nNk16 bf16, one accumulator coordinate, on bits: HopperBF16WgmmaDot16_v1, its finite part on the raw words "
        "(fp.tc_dot16's stand_ins=False)")
HopperBF16WgmmaDot16V3.opt_in = True  # type: ignore[attr-defined]
HopperE4m3QgmmaDot32 = definition(
    "HopperE4m3QgmmaDot32", 2,
    lambda C, acc, *ab: fp.tc_dot_e4m3(C, acc, list(ab[:32]), list(ab[32:]), groups=(32,), W=14, F=-139),
    (("acc", F32), ("a", A32_8), ("b", A32_8)), F32, word=P.HopperE4m3QgmmaDot32,
    doc="wgmma.m64nNk32 e4m3, one accumulator coordinate, on bits: HopperE4m3QgmmaDot32_v1")
HopperE4m3MmaV2Dot32 = definition(
    "HopperE4m3MmaV2Dot32", 2,
    lambda C, acc, *ab: fp.tc_dot_e4m3(C, acc, [ab[k] for k in MMAV2_E4M3_K_ORDER], [ab[32 + k] for k in MMAV2_E4M3_K_ORDER],
                                       groups=(16, 16), W=26, F=-133),
    (("acc", F32), ("a", A32_8), ("b", A32_8)), F32, word=P.HopperE4m3MmaV2Dot32,
    doc="Triton's MMAv2 e4m3 k32 step on sm_90 (two chained f16 mma.sync.m16n8k16 over k mod 4 in {0, 1} then {2, 3}), one "
        "accumulator coordinate, on bits: HopperE4m3MmaV2Dot32_v1")
BlackwellE4m3QmmaDot32 = definition(
    "BlackwellE4m3QmmaDot32", 2,
    lambda C, acc, *ab: fp.tc_dot_e4m3(C, acc, list(ab[:32]), list(ab[32:]), groups=(32,), W=26, F=-133),
    (("acc", F32), ("a", A32_8), ("b", A32_8)), F32, word=P.BlackwellE4m3QmmaDot32,
    doc="mma.sync.m16n8k32 e4m3 on sm_120, one accumulator coordinate, on bits: BlackwellE4m3QmmaDot32_v1")
BlackwellE5m2QmmaDot32 = definition(
    "BlackwellE5m2QmmaDot32", 2,
    lambda C, acc, *ab: fp.tc_dot_e5m2(C, acc, list(ab[:32]), list(ab[32:])),
    (("acc", F32), ("a", A32_8), ("b", A32_8)), F32, word=P.BlackwellE5m2QmmaDot32,
    doc="mma.sync.m16n8k32 e5m2 on sm_120, one accumulator coordinate, on bits: BlackwellE5m2QmmaDot32_v1")
BlackwellNvf4OmmaDot64 = definition(
    "BlackwellNvf4OmmaDot64", 2,
    lambda C, acc, *ab: fp4.step_total(C, acc, ab[:64], ab[64:128], ab[128:132], ab[132:136]),
    (("acc", F32), ("a", A64_4), ("b", A64_4), ("sa", A4_8), ("sb", A4_8)), F32, word=P.BlackwellNvf4OmmaDot64,
    doc="mma.sync kind::mxf4nvf4 with UE4M3 scales over blocks of 16, one accumulator coordinate, on bits: "
        "BlackwellNvf4OmmaDot64_v1")
BlackwellMxf4OmmaDot64 = definition(
    "BlackwellMxf4OmmaDot64", 2,
    lambda C, acc, *ab: fp4.step_total(C, acc, ab[:64], ab[64:128], ab[128:130], ab[130:132], ue8m0=True),
    (("acc", F32), ("a", A64_4), ("b", A64_4), ("sa", A2_8), ("sb", A2_8)), F32, word=P.BlackwellMxf4OmmaDot64,
    doc="mma.sync kind::mxf4 with UE8M0 scales over blocks of 32, one accumulator coordinate, on bits: BlackwellMxf4OmmaDot64_v1")
BlackwellS8ImmaDot32 = definition(
    "BlackwellS8ImmaDot32", 2,
    lambda C, acc, *ab: step_s8_bits(C, BLACKWELL_SM120_S8_M16N8K32, acc, list(ab[:32]), list(ab[32:])),
    (("acc", F32), ("a", A32_8), ("b", A32_8)), F32, word=P.BlackwellS8ImmaDot32,
    doc="mma.sync.m16n8k32.satfinite s32.s8.s8.s32 on sm_120, one accumulator coordinate, on bits: BlackwellS8ImmaDot32_v1")
S32ToF32Rn = definition("S32ToF32Rn", 2, fp.s32_to_f32, (("a", F32),), F32, word=P.S32ToF32Rn,
                        doc="cvt.rn.f32.s32 on bits: S32ToF32Rn_v1 (round to nearest even, zero +0)")

for _step in (AmpereBF16TcDot16, HopperBF16WgmmaDot16, HopperBF16WgmmaDot16V3, HopperE4m3QgmmaDot32, HopperE4m3MmaV2Dot32,
              BlackwellE4m3QmmaDot32, BlackwellE5m2QmmaDot32, BlackwellNvf4OmmaDot64, BlackwellMxf4OmmaDot64, BlackwellS8ImmaDot32):
    silicon_step(_step, _step.word.silicon)


def _zero_acc(B) -> Coll:
    space, idx, leaf = B.call(BL.constant(0)).refs[0]
    return Coll(F32, broadcast(space, idx, leaf, 32))


def _steps(B, S, x, w, k: int = 16):
    acc = _zero_acc(B)
    for s in range(S.K // k):
        acc = B.call(S.DOT, acc, x[k * s:k * s + k], w[k * s:k * s + k])
    return acc


@composite("DotBf16", 3, ["K", "DOT"], lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.K, BF16))), F32),
           conformance="exact-model-tested", refuse=G.DotBf16.refuse)
def DotBf16(B, S, x, w):
    """The f32 accumulator after K/16 `DOT` steps in ascending k from the zero accumulator, on bits."""
    return _steps(B, S, x, w)


@composite("GemmCoordinate", 3, ["K", "DOT"], lambda S: ((("x", Array(S.K, BF16)), ("wrow", Array(S.K, BF16))), BF16),
           conformance="exact-model-tested", refuse=G.GemmCoordinate.refuse)
def GemmCoordinate(B, S, x, wrow):
    """One output word of `x @ W^T` on bits: K/16 `DOT` steps in ascending k from the zero accumulator, then `F2fpBf16_v2`."""
    return B.call(F2fpBf16, _steps(B, S, x, wrow))


@composite("GemmCoordinatePad", 2, ["T", "K", "DOT"], lambda S: ((("x", Array(S.T, BF16)), ("wrow", Array(S.T, BF16))), BF16),
           conformance="exact-model-tested", refuse=G.GemmCoordinatePad.refuse)
def GemmCoordinatePad(B, S, x, wrow):
    """`GemmCoordinate_v3{K, DOT}` on T words of x and of W's row, each followed by K - T words of constant-0 bits."""
    if S.T == S.K:
        return B.call(bind(GemmCoordinate, K=S.K, DOT=S.DOT), x, wrow)
    space, idx, leaf = B.call(BL.constant(0)).refs[0]
    zeros = broadcast(space, idx, leaf, 16 * (S.K - S.T))

    def pad(a):
        return Coll(Array(S.K, BF16), concat([a.refs, zeros]))

    return B.call(bind(GemmCoordinate, K=S.K, DOT=S.DOT), pad(x), pad(wrow))


@composite("PvRow", 2, ["T", "K", "D", "DOT"],
           lambda S: ((("p", Array(S.T, BF16)), *((f"v{t}", Array(S.D, BF16)) for t in range(S.T))), Array(S.D, BF16)),
           conformance="exact-model-tested", refuse=G.PvRow.refuse)
def PvRow(B, S, p, *vs):
    """`GemmCoordinatePad_v2{T, K, DOT}` of p and each of the D columns of V's key rows v0..v{T-1}, on bits."""
    v = Coll(Array(S.T, Array(S.D, BF16)), concat([r.refs for r in vs]))
    return B.batch(bind(GemmCoordinatePad, T=S.T, K=S.K, DOT=S.DOT), p, v, axes=(None, 1))


@composite("PvRowPad", 2, ["T", "K", "D", "DOT"],
           lambda S: ((("p", Array(S.T, BF16)), *((f"v{t}", Array(S.D, BF16)) for t in range(S.T)), ("n", Array(16, BIT))),
                      Array(S.D, BF16)),
           conformance="exact-model-tested", refuse=G.PvRowPad.refuse, ports=G.KEY_COUNT)
def PvRowPad(B, S, p, *rest):
    """`PvRow_v2{T, K, D, DOT}` on bits with key t's p word and V row kept when t < n, unsigned (`PvRowPad_v1`'s
    compare), and constant-0 bits otherwise (`SelectBf16_v2`)."""
    *vs, n = rest
    space, idx, leaf = B.call(BL.constant(0)).refs[0]
    zero = Coll(BF16, broadcast(space, idx, leaf, 16))
    count = B.call(EW.I32Add, B.call(SC.Bf16ToF32, n), B.call(SC.const_word(32, 1 << 31)))
    keep = [B.call(SC.I32Le, B.call(SC.const_word(32, ((t + 1) << 16) ^ (1 << 31))), count) for t in range(S.T)]
    ps = Coll(Array(S.T, BF16), concat([B.call(SC.SelectBf16, keep[t], p[t], zero).refs for t in range(S.T)]))
    rows = [B.batch(SC.SelectBf16, keep[t], v, zero, axes=(None, 0, None)) for t, v in enumerate(vs)]
    return B.call(bind(PvRow, T=S.T, K=S.K, D=S.D, DOT=S.DOT), ps, *rows)


@composite("Gemm", 3, ["K", "N", "DOT"],
           lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, BF16)))), Array(S.N, BF16)),
           conformance="exact-model-tested", refuse=G.Gemm.refuse)
def Gemm(B, S, x, w):
    """`x[K] @ W[N,K]^T -> [N]` on bits, no epilogue: `GemmCoordinate_v3` over W's rows."""
    return B.batch(bind(GemmCoordinate, K=S.K, DOT=S.DOT), x, w, axes=(None, 0))


@composite("GemmCoordinateE4m3", 2, ["K", "DOT"], lambda S: ((("x", Array(S.K, E4M3)), ("wrow", Array(S.K, E4M3))), BF16),
           conformance="exact-model-tested", refuse=G.GemmCoordinateE4m3.refuse)
def GemmCoordinateE4m3(B, S, x, wrow):
    """One output word of an E4M3 `x @ W^T` on bits: K/32 `DOT` steps in ascending k from the zero accumulator, then
    `F2fpBf16_v2`."""
    return B.call(F2fpBf16, _steps(B, S, x, wrow, 32))


def _block_scaled_coordinate(B, S, x, wrow, sx, sw, ns: int):
    acc = _zero_acc(B)
    for s in range(S.K // 64):
        acc = B.call(S.DOT, acc, x[64 * s:64 * s + 64], wrow[64 * s:64 * s + 64], sx[ns * s:ns * s + ns], sw[ns * s:ns * s + ns])
    return B.call(F2fpBf16, acc)


@composite("GemmCoordinateNvf4", 2, ["K", "DOT"],
           lambda S: ((("x", Array(S.K, E2M1)), ("wrow", Array(S.K, E2M1)), ("sx", Array(S.K // 16, SCALE)),
                       ("sw", Array(S.K // 16, SCALE))), BF16),
           conformance="exact-model-tested", refuse=G.GemmCoordinateNvf4.refuse)
def GemmCoordinateNvf4(B, S, x, wrow, sx, sw):
    """One output word of an NVFP4 `x @ W^T` on bits (one UE4M3 scale byte per 16 codes): K/64 `DOT` steps in ascending k
    from the zero accumulator, then `F2fpBf16_v2`."""
    return _block_scaled_coordinate(B, S, x, wrow, sx, sw, 4)


@composite("GemmCoordinateMxf4", 2, ["K", "DOT"],
           lambda S: ((("x", Array(S.K, E2M1)), ("wrow", Array(S.K, E2M1)), ("sx", Array(S.K // 32, SCALE)),
                       ("sw", Array(S.K // 32, SCALE))), BF16),
           conformance="exact-model-tested", refuse=G.GemmCoordinateMxf4.refuse)
def GemmCoordinateMxf4(B, S, x, wrow, sx, sw):
    """One output word of an MXFP4 `x @ W^T` on bits (one UE8M0 scale byte per 32 codes): K/64 `DOT` steps in ascending k
    from the zero accumulator, then `F2fpBf16_v2`."""
    return _block_scaled_coordinate(B, S, x, wrow, sx, sw, 2)


@composite("GemmAccumulatorS8", 2, ["K", "DOT"], lambda S: ((("x", Array(S.K, E4M3)), ("wrow", Array(S.K, E4M3))), F32),
           conformance="exact-model-tested", refuse=G.GemmAccumulatorS8.refuse)
def GemmAccumulatorS8(B, S, x, wrow):
    """The s32 accumulator of an s8 `x @ W^T` coordinate on bits: K/32 `DOT` steps in ascending k from the zero accumulator."""
    return _steps(B, S, x, wrow, 32)


GemmAccumulatorS8.counts = FAMILY_COUNTS["GemmAccumulatorS8_v2"] = serves_none  # type: ignore[attr-defined]


@composite("ScaledMmS8Coordinate", 2, ["K", "DOT"],
           lambda S: ((("x", Array(S.K, E4M3)), ("sx", F32), ("wrow", Array(S.K, E4M3)), ("sw", F32)), BF16),
           conformance=G.ScaledMmS8Coordinate.conformance, refuse=G.ScaledMmS8Coordinate.refuse)
def ScaledMmS8Coordinate(B, S, x, sx, wrow, sw):
    """One output word of a W8A8 linear on bits: `F2fpBf16_v2((S32ToF32Rn_v2(acc) * sx) * sw)` with `F32Mul_v4`."""
    acc = B.call(S32ToF32Rn, B.call(bind(GemmAccumulatorS8, K=S.K, DOT=S.DOT), x, wrow))
    return B.call(F2fpBf16, B.call(EW.F32MulV4, B.call(EW.F32MulV4, acc, sx), sw))


@composite("ScaledMmS8CoordinateRows", 2, ["K", "R", "DOT"],
           lambda S: ((*((f"x{r}", Array(S.K // S.R, E4M3)) for r in range(S.R)), ("sx", F32),
                       *((f"wrow{r}", Array(S.K // S.R, E4M3)) for r in range(S.R)), ("sw", F32)), BF16),
           conformance=G.ScaledMmS8CoordinateRows.conformance, refuse=G.ScaledMmS8CoordinateRows.refuse)
def ScaledMmS8CoordinateRows(B, S, *ports):
    """`ScaledMmS8Coordinate_v2{K, DOT}` on bits, on the concatenations of x's and W's row's R parts."""
    R = S.R

    def cat(parts):
        return Coll(Array(S.K, E4M3), concat([p.refs for p in parts]))

    x, sx, wrow, sw = cat(ports[:R]), ports[R], cat(ports[R + 1:2 * R + 1]), ports[2 * R + 1]
    return B.call(bind(ScaledMmS8Coordinate, K=S.K, DOT=S.DOT), x, sx, wrow, sw)


@composite("ScaledMmS8ResidualCoordinateRows", 2, ["K", "R", "DOT"],
           lambda S: ((*ScaledMmS8CoordinateRows.signature(S)[0], ("r", BF16)), BF16),
           conformance=G.ScaledMmS8ResidualCoordinateRows.conformance, refuse=G.ScaledMmS8ResidualCoordinateRows.refuse)
def ScaledMmS8ResidualCoordinateRows(B, S, *ports):
    """`ScaledMmS8ResidualCoordinateRows_v1`'s body on bits: `Bf16AddF2fp_v2(r, ScaledMmS8CoordinateRows_v2{K, R, DOT})`."""
    y = B.call(bind(ScaledMmS8CoordinateRows, K=S.K, R=S.R, DOT=S.DOT), *ports[:-1])
    return B.call(EW.Bf16AddF2fp, ports[-1], y)


@composite("ScaledMmS8", 2, ["K", "N", "DOT"],
           lambda S: ((("x", Array(S.K, E4M3)), ("sx", F32), ("w", Array(S.N, Array(S.K, E4M3))), ("sw", Array(S.N, F32))),
                      Array(S.N, BF16)),
           conformance=G.ScaledMmS8.conformance, refuse=G.ScaledMmS8.refuse)
def ScaledMmS8(B, S, x, sx, w, sw):
    """A W8A8 linear on one token row on bits: `ScaledMmS8Coordinate_v2` over the N weight rows and their scales."""
    return B.batch(bind(ScaledMmS8Coordinate, K=S.K, DOT=S.DOT), x, sx, w, sw, axes=(None, None, 0, 0))


DotBf16.word = lambda b: bind(G.DotBf16, K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
GemmCoordinate.word = lambda b: bind(G.GemmCoordinate, K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
GemmCoordinatePad.word = lambda b: bind(G.GemmCoordinatePad, T=b["T"], K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
PvRow.word = lambda b: bind(G.PvRow, T=b["T"], K=b["K"], D=b["D"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
PvRowPad.word = lambda b: bind(G.PvRowPad, T=b["T"], K=b["K"], D=b["D"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
Gemm.word = lambda b: bind(G.Gemm, K=b["K"], N=b["N"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
GemmCoordinateE4m3.word = lambda b: bind(G.GemmCoordinateE4m3, K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
GemmCoordinateNvf4.word = lambda b: bind(G.GemmCoordinateNvf4, K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
GemmCoordinateMxf4.word = lambda b: bind(G.GemmCoordinateMxf4, K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
GemmAccumulatorS8.word = lambda b: bind(G.GemmAccumulatorS8, K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
ScaledMmS8Coordinate.word = lambda b: bind(G.ScaledMmS8Coordinate, K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
ScaledMmS8CoordinateRows.word = lambda b: bind(G.ScaledMmS8CoordinateRows, K=b["K"], R=b["R"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
ScaledMmS8ResidualCoordinateRows.word = lambda b: bind(G.ScaledMmS8ResidualCoordinateRows, K=b["K"], R=b["R"],  # type: ignore[attr-defined]
                                                       DOT=word_of(b["DOT"]))
ScaledMmS8.word = lambda b: bind(G.ScaledMmS8, K=b["K"], N=b["N"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]

GEMM_AMPERE: dict = {"DOT": AmpereBF16TcDot16}
GEMM_HOPPER: dict = {"DOT": HopperBF16WgmmaDot16}
GEMM_HOPPER_E4M3: dict = {"DOT": HopperE4m3QgmmaDot32}
GEMM_HOPPER_E4M3_MMAV2: dict = {"DOT": HopperE4m3MmaV2Dot32}
GEMM_SM120_E4M3: dict = {"DOT": BlackwellE4m3QmmaDot32}
GEMM_SM120_NVF4: dict = {"DOT": BlackwellNvf4OmmaDot64}
GEMM_SM120_MXF4: dict = {"DOT": BlackwellMxf4OmmaDot64}
GEMM_SM120_S8: dict = {"DOT": BlackwellS8ImmaDot32}


def s8_gemm(model) -> dict:
    """`G.s8_gemm` on bits: `GEMM_SM120_S8`'s step on bits for a card whose s8 step is ``model``, refused as `G.s8_gemm`
    refuses."""
    G.s8_gemm(model)
    return dict(GEMM_SM120_S8)


#: sm_90's on bits (`G.GEMM_SM90_S8`)
GEMM_SM90_S8: dict = s8_gemm(HOPPER_SM90_S8_K32)
