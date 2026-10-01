"""The BF16 GEMM coordinate as Boolean Definitions: the next version of each word Definition, the same function on bits.

    F2fpBf16_v2                 (a: Array<32, Value<1>>) -> Array<16, Value<1>>                       word view: F2fpBf16_v1
    AmpereBF16TcDot16_v3        (acc: Array<32, Value<1>>, a, b: Array<16, Array<16, Value<1>>>)
                                -> Array<32, Value<1>>                                                word view: AmpereBF16TcDot16_v2
    HopperBF16WgmmaDot16_v2     the same signature                                                    word view: HopperBF16WgmmaDot16_v1
    GemmCoordinate_v3{K, DOT}   (x, wrow: Array<K, Array<16, Value<1>>>) -> Array<16, Value<1>>       word view: GemmCoordinate_v2{K, DOT'}

The steps and the cast are traced from `fp`'s builders (`trace.definition`): their bodies are And/Xor/Not gates, with exactly
the builders' live ANDs. `GemmCoordinate_v3` is `GemmCoordinate_v2`'s body on bits: K/16 calls of `DOT` from the zero
accumulator (32 constant-0 bits), then `F2fpBf16_v2`. `word_of` names each one's word Definition (``DOT'`` is `DOT`'s);
the tests check that the two agree, on random and special words and on the hardware captures.
"""
from __future__ import annotations

from verity.ir import boolean as BL
from verity.ir.defs import CompositeDefinition, SpecializedDefinition, bind, composite
from verity.ir.refs import Coll, broadcast
from verity.ir.types import Array

from .. import gemm as G
from .. import prims as P
from . import fp
from .trace import definition

__all__ = ["AmpereBF16TcDot16", "F2fpBf16", "GEMM_AMPERE", "GEMM_HOPPER", "GemmCoordinate", "HopperBF16WgmmaDot16", "word_of"]

BIT = BL.BIT
F32 = Array(32, BIT)
BF16 = Array(16, BIT)
A16 = Array(16, BF16)

#: F2fpBf16_v1's NaN: every NaN input gives 0x7FFF
NAN_F2FP = 0x7FFF

F2fpBf16 = definition("F2fpBf16", 2, lambda C, a: fp.f32_to_bf16(C, a, NAN_F2FP), (("a", F32),), BF16,
                      doc="cvt.rn.bf16.f32 on bits: F2fpBf16_v1 (round to nearest even, NaN -> 0x7FFF)")
AmpereBF16TcDot16 = definition(
    "AmpereBF16TcDot16", 3, lambda C, acc, *ab: fp.tc_dot16(C, acc, list(ab[:16]), list(ab[16:])),
    (("acc", F32), ("a", A16), ("b", A16)), F32,
    doc="mma.sync.m16n8k16 bf16, one accumulator coordinate, on bits: AmpereBF16TcDot16_v2")
HopperBF16WgmmaDot16 = definition(
    "HopperBF16WgmmaDot16", 2,
    lambda C, acc, *ab: fp.tc_dot16(C, acc, list(ab[:16]), list(ab[16:]), groups=(16,), W=26, F=-133),
    (("acc", F32), ("a", A16), ("b", A16)), F32,
    doc="wgmma.m64nNk16 bf16, one accumulator coordinate, on bits: HopperBF16WgmmaDot16_v1")


@composite("GemmCoordinate", 3, ["K", "DOT"], lambda S: ((("x", Array(S.K, BF16)), ("wrow", Array(S.K, BF16))), BF16),
           conformance="exact-model-tested")
def GemmCoordinate(B, S, x, wrow):
    """One output word of `x @ W^T` on bits: K/16 `DOT` steps in ascending k from the zero accumulator, then `F2fpBf16_v2`."""
    zero = B.call(BL.constant(0))
    space, idx, leaf = zero.refs[0]
    acc = Coll(F32, broadcast(space, idx, leaf, 32))
    for s in range(S.K // 16):
        acc = B.call(S.DOT, acc, x[16 * s:16 * s + 16], wrow[16 * s:16 * s + 16])
    return B.call(F2fpBf16, acc)


GEMM_AMPERE: dict = {"DOT": AmpereBF16TcDot16}
GEMM_HOPPER: dict = {"DOT": HopperBF16WgmmaDot16}

_WORD = {F2fpBf16: P.F2fpBf16, AmpereBF16TcDot16: P.AmpereBF16TcDot16, HopperBF16WgmmaDot16: P.HopperBF16WgmmaDot16}


def word_of(target):
    """The word Definition a Boolean Definition of this module computes (through `boolean.word_type`)."""
    if isinstance(target, SpecializedDefinition) and target.defn is GemmCoordinate:
        return bind(G.GemmCoordinate, K=target.bindings["K"], DOT=word_of(target.bindings["DOT"]))
    if isinstance(target, SpecializedDefinition) and not target.defn.statics:
        target = target.defn
    if isinstance(target, CompositeDefinition) and target in _WORD:
        return _WORD[target]
    raise KeyError(f"{getattr(target, 'id', target)!r} has no word counterpart here")
