"""The BF16 GEMM coordinate as Boolean Definitions: the next version of each word Definition, the same function on bits.

    F2fpBf16_v2                 (a: Array<32, Value<1>>) -> Array<16, Value<1>>                       word view: F2fpBf16_v1
    AmpereBF16TcDot16_v3        (acc: Array<32, Value<1>>, a, b: Array<16, Array<16, Value<1>>>)
                                -> Array<32, Value<1>>                                                word view: AmpereBF16TcDot16_v2
    HopperBF16WgmmaDot16_v2     the same signature                                                    word view: HopperBF16WgmmaDot16_v1
    DotBf16_v3{K, DOT}          (x, w: Array<K, Array<16, Value<1>>>) -> Array<32, Value<1>>          word view: DotBf16_v2{K, DOT'}
    GemmCoordinate_v3{K, DOT}   (x, wrow: Array<K, Array<16, Value<1>>>) -> Array<16, Value<1>>       word view: GemmCoordinate_v2{K, DOT'}
    Gemm_v3{K, N, DOT}          (x: Array<K, bf16 bits>, w: Array<N, Array<K, bf16 bits>>)
                                -> Array<N, Array<16, Value<1>>>                                      word view: Gemm_v2{K, N, DOT'}

The steps and the cast are traced from `fp`'s builders (`trace.definition`): their bodies are And/Xor/Not gates, with exactly
the builders' live ANDs. `GemmCoordinate_v3` is `GemmCoordinate_v2`'s body on bits: K/16 calls of `DOT` from the zero
accumulator (32 constant-0 bits), then `F2fpBf16_v2`; `DotBf16_v3` is the same steps without the cast, and `Gemm_v3` is
`Gemm_v2`'s: the coordinate batched over W's N rows. Each names its word
Definition (`trace.word_of`; ``DOT'`` is `DOT`'s); the tests check that the two agree, on random and special words and on the
hardware captures.
"""
from __future__ import annotations

from verity.ir import boolean as BL
from verity.ir.defs import bind, composite
from verity.ir.refs import Coll, broadcast
from verity.ir.types import Array

from .. import gemm as G
from .. import prims as P
from . import fp
from .trace import definition, word_of

__all__ = ["AmpereBF16TcDot16", "DotBf16", "F2fpBf16", "GEMM_AMPERE", "GEMM_HOPPER", "Gemm", "GemmCoordinate",
           "HopperBF16WgmmaDot16"]

BIT = BL.BIT
F32 = Array(32, BIT)
BF16 = Array(16, BIT)
A16 = Array(16, BF16)

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


def _steps(B, S, x, w):
    zero = B.call(BL.constant(0))
    space, idx, leaf = zero.refs[0]
    acc = Coll(F32, broadcast(space, idx, leaf, 32))
    for s in range(S.K // 16):
        acc = B.call(S.DOT, acc, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
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


DotBf16.word = lambda b: bind(G.DotBf16, K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
GemmCoordinate.word = lambda b: bind(G.GemmCoordinate, K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
Gemm.word = lambda b: bind(G.Gemm, K=b["K"], N=b["N"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]

GEMM_AMPERE: dict = {"DOT": AmpereBF16TcDot16}
GEMM_HOPPER: dict = {"DOT": HopperBF16WgmmaDot16}
