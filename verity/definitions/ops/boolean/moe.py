"""The routed expert GEMMs of a padded static MoE layer as Boolean Definitions: the next version of each `ops.moe` word
Definition, the same function on bits.

    MoeSelectBf16x{E}_v2              (e: Array<32, Value<1>>, column: Array<E, Array<16, Value<1>>>) -> Array<16, Value<1>>
                                      word view: MoeSelectBf16x{E}_v1
    MoePairCoordinate_v2{E, K, DOT}   (x: Array<K, bf16 bits>, e: Array<32, Value<1>>, slab: Array<E, Array<K, bf16 bits>>)
                                      -> Array<16, Value<1>>                                     word view: MoePairCoordinate_v1{E, K, DOT'}
    MoePairCoordinateW_v2{E, K, DOT}  (x, e, g: Array<32, Value<1>>, slab) -> Array<16, Value<1>> word view: MoePairCoordinateW_v1{E, K, DOT'}
    MoePairGemm_v2{E, K, N, DOT}      (x, e, w: Array<E, Array<N, Array<K, bf16 bits>>>) -> Array<N, Array<16, Value<1>>>
                                                                                                 word view: MoePairGemm_v1{E, K, N, DOT'}
    MoePairGemmW_v2{E, K, N, DOT}     (x, e, g, w) -> Array<N, Array<16, Value<1>>>                word view: MoePairGemmW_v1{E, K, N, DOT'}

`MoeSelectBf16x{E}_v2` is traced (`trace.definition`): the multiplexer tree (`boolean.gather`) over the id's low b =
ceil(log2 E) bits, a word past E standing for 0x7FFF, then one mux per output bit to 0x7FFF when any of the id's other 32 - b
bits is set (so a negative id, whose sign bit is set, selects 0x7FFF too): (E - 1 + p) * 16 + 16 + 31 - b ANDs, p the levels
at which an odd count meets the fill; 2,072 at E = 128 and 4,119 at E = 256. Every AND of the tree reads a candidate's bits or
a partial result's, so selects by one id over different columns share no gate but the id's range bit.

Each pair composite is its word body on bits: one select per k over the column's E experts, then `gemm`'s
`GemmCoordinate_v3{K, DOT}`, or `DotBf16_v3{K, DOT}`, `elementwise`'s `F32Mul_v4` and `F2fpBf16_v2`; `DOT` a Boolean k16
step, read as its word view (``DOT'``). Each refuses at `bind` the K its word Definition refuses.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import const_bits, mux, or_reduce
from verity.ml.circuits.boolean.gather import gather
from verity.ml.circuits.boolean.trace import definition, word_of
from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.types import Array

from .. import gemm as G
from .. import moe as W
from . import elementwise as EW
from . import gemm as BG

__all__ = ["MoePairCoordinate", "MoePairCoordinateW", "MoePairGemm", "MoePairGemmW", "SELECT_LIMIT", "select_bf16"]

BIT = BL.BIT
BF16 = Array(16, BIT)
F32 = I32 = Array(32, BIT)
#: the most experts a traced select chooses among: its tree is (E - 1) * 16 ANDs in one Definition
SELECT_LIMIT = 1024

_SELECTS: dict[int, Any] = {}


def _select(E: int):
    b = (E - 1).bit_length()

    def build(C, e, *column):
        lo = gather(C, e[:b], list(column), W.FILL)
        return mux(C, or_reduce(C, e[b:]), const_bits(W.FILL, 16), lo)

    return build


def select_bf16(E: int) -> Any:
    """`MoeSelectBf16x{E}_v2`: `MoeSelectBf16x{E}_v1` on bits."""
    if not 1 <= E <= SELECT_LIMIT:
        raise ValueError(f"a select on bits chooses among 1 to {SELECT_LIMIT} experts, not {E}")
    d = _SELECTS.get(E)
    if d is None:
        d = _SELECTS[E] = definition(
            f"MoeSelectBf16x{E}", 2, _select(E), (("e", I32), ("column", Array(E, BF16))), BF16, word=W.select_bf16(E),
            doc=f"MoeSelectBf16x{E}_v1 on bits: column[e] for the signed i32 e in [0, {E}), else 0x7FFF; the multiplexer tree "
                f"over e's low {(E - 1).bit_length()} bits, then 0x7FFF where any higher bit is set")
    return d


def _coordinate_sig(routed: bool):
    g = (("g", F32),) if routed else ()
    return lambda S: ((("x", Array(S.K, BF16)), ("e", I32)) + g + (("slab", Array(S.E, Array(S.K, BF16))),), BF16)


def _gemm_sig(routed: bool):
    g = (("g", F32),) if routed else ()
    return lambda S: ((("x", Array(S.K, BF16)), ("e", I32)) + g + (("w", Array(S.E, Array(S.N, Array(S.K, BF16)))),),
                      Array(S.N, BF16))


def _row(B, S, e, slab):
    return B.batch(select_bf16(S.E), e, W.columns(slab, S.E, S.K, BF16), axes=(None, 0))


@composite("MoePairCoordinate", 2, ["E", "K", "DOT"], _coordinate_sig(False), conformance="exact-model-tested",
           refuse=G.GemmCoordinate.refuse)
def MoePairCoordinate(B, S, x, e, slab):
    """`MoePairCoordinate_v1`'s body on bits: K `MoeSelectBf16x{E}_v2`, then `GemmCoordinate_v3{K, DOT}`."""
    return B.call(bind(BG.GemmCoordinate, K=S.K, DOT=S.DOT), x, _row(B, S, e, slab))


@composite("MoePairCoordinateW", 2, ["E", "K", "DOT"], _coordinate_sig(True), conformance="exact-model-tested",
           refuse=G.GemmCoordinate.refuse)
def MoePairCoordinateW(B, S, x, e, g, slab):
    """`MoePairCoordinateW_v1`'s body on bits: K `MoeSelectBf16x{E}_v2`, `DotBf16_v3{K, DOT}`, `F32Mul_v4` by g, then
    `F2fpBf16_v2`."""
    acc = B.call(bind(BG.DotBf16, K=S.K, DOT=S.DOT), x, _row(B, S, e, slab))
    return B.call(BG.F2fpBf16, B.call(EW.F32MulV4, acc, g))


@composite("MoePairGemm", 2, ["E", "K", "N", "DOT"], _gemm_sig(False), conformance="exact-model-tested",
           refuse=G.GemmCoordinate.refuse)
def MoePairGemm(B, S, x, e, w):
    """`MoePairGemm_v1`'s body on bits: `MoePairCoordinate_v2` over each output coordinate's [E][K] slab."""
    return B.batch(bind(MoePairCoordinate, E=S.E, K=S.K, DOT=S.DOT), x, e, W.coordinate_major(w, S.E, S.N, S.K, BF16),
                   axes=(None, None, 0))


@composite("MoePairGemmW", 2, ["E", "K", "N", "DOT"], _gemm_sig(True), conformance="exact-model-tested",
           refuse=G.GemmCoordinate.refuse)
def MoePairGemmW(B, S, x, e, g, w):
    """`MoePairGemmW_v1`'s body on bits: `MoePairCoordinateW_v2` over each output coordinate's [E][K] slab."""
    return B.batch(bind(MoePairCoordinateW, E=S.E, K=S.K, DOT=S.DOT), x, e, g, W.coordinate_major(w, S.E, S.N, S.K, BF16),
                   axes=(None, None, None, 0))


MoePairCoordinate.word = lambda b: bind(W.MoePairCoordinate, E=b["E"], K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
MoePairCoordinateW.word = lambda b: bind(W.MoePairCoordinateW, E=b["E"], K=b["K"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
MoePairGemm.word = lambda b: bind(W.MoePairGemm, E=b["E"], K=b["K"], N=b["N"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
MoePairGemmW.word = lambda b: bind(W.MoePairGemmW, E=b["E"], K=b["K"], N=b["N"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
