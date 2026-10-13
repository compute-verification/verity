"""The BF16 GEMM coordinate as Boolean Definitions: the next version of each word Definition, the same function on bits.

    F2fpBf16_v2                 (a: Array<32, Value<1>>) -> Array<16, Value<1>>                       word view: F2fpBf16_v1
    AmpereBF16TcDot16_v3        (acc: Array<32, Value<1>>, a, b: Array<16, Array<16, Value<1>>>)
                                -> Array<32, Value<1>>                                                word view: AmpereBF16TcDot16_v2
    HopperBF16WgmmaDot16_v2     the same signature                                                    word view: HopperBF16WgmmaDot16_v1
    DotBf16_v3{K, DOT}          (x, w: Array<K, Array<16, Value<1>>>) -> Array<32, Value<1>>          word view: DotBf16_v2{K, DOT'}
    GemmCoordinate_v3{K, DOT}   (x, wrow: Array<K, Array<16, Value<1>>>) -> Array<16, Value<1>>       word view: GemmCoordinate_v2{K, DOT'}
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

The steps and the cast are traced from `fp`'s builders (`trace.definition`): their bodies are And/Xor/Not gates, with exactly
the builders' live ANDs. `GemmCoordinate_v3` is `GemmCoordinate_v2`'s body on bits: K/16 calls of `DOT` from the zero
accumulator (32 constant-0 bits), then `F2fpBf16_v2`; `DotBf16_v3` is the same steps without the cast, and `Gemm_v3` is
`Gemm_v2`'s: the coordinate batched over W's N rows. The E4M3 steps are `fp.tc_dot_e4m3` at each step's groups, adder width and floor
(`HopperE4m3MmaV2Dot32_v2` on the operands in `MMAV2_E4M3_K_ORDER`; the E5M2 step `fp.tc_dot_e5m2`),
and `GemmCoordinateE4m3_v2` is K/32 of them, then the cast; the FP4 steps are `fp4.step_total` and their coordinates K/64 of
them. Each names its word
Definition (`trace.word_of`; ``DOT'`` is `DOT`'s); the tests check that the two agree, on random and special words and on the
hardware captures.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Coll, broadcast
from verity.ml.circuits.types import Array

from .. import gemm as G
from .. import prims as P
from verity.ml.circuits.silicon import fp, fp4
from verity.ml.circuits.silicon.total_fp8 import MMAV2_E4M3_K_ORDER
from verity.ml.circuits.boolean.trace import definition, word_of

__all__ = ["AmpereBF16TcDot16", "BlackwellE4m3QmmaDot32", "BlackwellE5m2QmmaDot32", "BlackwellMxf4OmmaDot64", "BlackwellNvf4OmmaDot64", "DotBf16", "F2fpBf16",
           "GEMM_AMPERE", "GEMM_HOPPER", "GEMM_HOPPER_E4M3", "GEMM_HOPPER_E4M3_MMAV2", "GEMM_SM120_E4M3", "GEMM_SM120_MXF4", "GEMM_SM120_NVF4", "Gemm",
           "GemmCoordinate", "GemmCoordinateE4m3", "GemmCoordinateMxf4", "GemmCoordinateNvf4", "HopperBF16WgmmaDot16",
           "HopperE4m3MmaV2Dot32", "HopperE4m3QgmmaDot32"]

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


def _zero_acc(B) -> Coll:
    space, idx, leaf = B.call(BL.constant(0)).refs[0]
    return Coll(F32, broadcast(space, idx, leaf, 32))


def _steps(B, S, x, w, k: int = 16):
    acc = _zero_acc(B)
    for s in range(S.K // k):
        acc = B.call(S.DOT, acc, x[k * s:k * s + k], w[k * s:k * s + k])
    return acc


@composite("DotBf16", 3, ["K", "DOT"], lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.K, BF16))), F32),
           conformance="exact-model-tested")
def DotBf16(B, S, x, w):
    """The f32 accumulator after K/16 `DOT` steps in ascending k from the zero accumulator, on bits."""
    return _steps(B, S, x, w)


@composite("GemmCoordinate", 3, ["K", "DOT"], lambda S: ((("x", Array(S.K, BF16)), ("wrow", Array(S.K, BF16))), BF16),
           conformance="exact-model-tested")
def GemmCoordinate(B, S, x, wrow):
    """One output word of `x @ W^T` on bits: K/16 `DOT` steps in ascending k from the zero accumulator, then `F2fpBf16_v2`."""
    return B.call(F2fpBf16, _steps(B, S, x, wrow))


@composite("Gemm", 3, ["K", "N", "DOT"],
           lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, BF16)))), Array(S.N, BF16)),
           conformance="exact-model-tested")
def Gemm(B, S, x, w):
    """`x[K] @ W[N,K]^T -> [N]` on bits, no epilogue: `GemmCoordinate_v3` over W's rows."""
    return B.batch(bind(GemmCoordinate, K=S.K, DOT=S.DOT), x, w, axes=(None, 0))


@composite("GemmCoordinateE4m3", 2, ["K", "DOT"], lambda S: ((("x", Array(S.K, E4M3)), ("wrow", Array(S.K, E4M3))), BF16),
           conformance="exact-model-tested")
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
           conformance="exact-model-tested")
def GemmCoordinateNvf4(B, S, x, wrow, sx, sw):
    """One output word of an NVFP4 `x @ W^T` on bits (one UE4M3 scale byte per 16 codes): K/64 `DOT` steps in ascending k
    from the zero accumulator, then `F2fpBf16_v2`."""
    return _block_scaled_coordinate(B, S, x, wrow, sx, sw, 4)


@composite("GemmCoordinateMxf4", 2, ["K", "DOT"],
           lambda S: ((("x", Array(S.K, E2M1)), ("wrow", Array(S.K, E2M1)), ("sx", Array(S.K // 32, SCALE)),
                       ("sw", Array(S.K // 32, SCALE))), BF16),
           conformance="exact-model-tested")
def GemmCoordinateMxf4(B, S, x, wrow, sx, sw):
    """One output word of an MXFP4 `x @ W^T` on bits (one UE8M0 scale byte per 32 codes): K/64 `DOT` steps in ascending k
    from the zero accumulator, then `F2fpBf16_v2`."""
    return _block_scaled_coordinate(B, S, x, wrow, sx, sw, 2)


DotBf16.word = lambda b: bind(G.DotBf16, K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
GemmCoordinate.word = lambda b: bind(G.GemmCoordinate, K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
Gemm.word = lambda b: bind(G.Gemm, K=b["K"], N=b["N"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
GemmCoordinateE4m3.word = lambda b: bind(G.GemmCoordinateE4m3, K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
GemmCoordinateNvf4.word = lambda b: bind(G.GemmCoordinateNvf4, K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
GemmCoordinateMxf4.word = lambda b: bind(G.GemmCoordinateMxf4, K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]

GEMM_AMPERE: dict = {"DOT": AmpereBF16TcDot16}
GEMM_HOPPER: dict = {"DOT": HopperBF16WgmmaDot16}
GEMM_HOPPER_E4M3: dict = {"DOT": HopperE4m3QgmmaDot32}
GEMM_HOPPER_E4M3_MMAV2: dict = {"DOT": HopperE4m3MmaV2Dot32}
GEMM_SM120_E4M3: dict = {"DOT": BlackwellE4m3QmmaDot32}
GEMM_SM120_NVF4: dict = {"DOT": BlackwellNvf4OmmaDot64}
GEMM_SM120_MXF4: dict = {"DOT": BlackwellMxf4OmmaDot64}
