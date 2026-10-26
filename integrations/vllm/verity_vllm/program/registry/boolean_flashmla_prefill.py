"""`flashmla_prefill`'s DeepSeek-V4-Pro sparse MLA prefill on bits: version 2 of each Definition, its word body line for line over
Boolean Definitions (the bodies are `flashmla_prefill`'s, written once over `flashmla.Ops`), so each one's word view is its `_v1` at
the same statics with the Boolean ``DOT`` read as its word view.

    E8m0ShlBf16_v2                                   wires only: the byte into bits 7..14
    FlashMlaGatherRow_v2{DN,DR,QT}
    FlashMlaPrefillValid_v2                          traced: idx's sign bit and two borrows
    FlashMlaPrefillPair_v2{D,BN,FIRST,DOT,SCALE}     DOT: `gemm.HopperBF16WgmmaDot16_v2` (the prefill's wgmma k16 step)
    FlashMlaPrefillHead_v2{NP,D,BN,DOT,SCALE}
    FlashMlaPrefill_v2{NH,NP,D,BN,DOT,SCALE}

The callees are `boolean_flashmla`'s steps (core's Boolean pieces and the decode's libm pieces) and `F32Div_v4` for the divide.  No
table or lookup gate.  Each composite is one traced circuit (`boolean_dsa.flat`), so a row's scales and the query every score
reads are decoded once.
"""
from __future__ import annotations

from verity.ml.circuits.boolean.forms import NOT, ZERO, and_reduce, sub
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import gemm as BG
from verity_vllm.program.registry import boolean_flashmla as BF
from verity_vllm.program.registry import flashmla_prefill as P
from verity_vllm.program.registry.boolean_dsa import flat

BIT, BF16, I32, E8, BIT_T = BF.BIT, BF.BF16, BF.I32, BF.E8, BF.BIT_T
_LIFTED = BF._LIFTED

E8m0ShlBf16V2 = definition("E8m0ShlBf16", 2, lambda C, x: [ZERO] * 7 + list(x) + [ZERO], (("x", E8),), BF16, word=P.E8m0ShlBf16,
                           doc="the prefill gather's scale word on bits: E8m0ShlBf16_v1, x << 7")

BITS = {**BF.BITS, "div": EW.F32DivV4, "shl": E8m0ShlBf16V2}

#: DeepSeek-V4-Pro at TP8 on bits: `flashmla_prefill.FLASHMLA_PREFILL_DSV4` with the Boolean DOT
FLASHMLA_PREFILL_DSV4 = {**P.FLASHMLA_PREFILL_DSV4, "DOT": BG.HopperBF16WgmmaDot16}


def bit_ops(B):
    return BF.bit_ops(B, BITS)


def _gather(B, S, nope, scale, rope):
    """`FlashMlaGatherRow_v1`'s body on bits."""
    return P.gather_row(bit_ops(B), S, nope, scale, rope)


FlashMlaGatherRowV2 = flat("FlashMlaGatherRow", 2, ["DN", "DR", "QT"], lambda S: P._gather_type(S, BIT_T), _gather,
                           P.FlashMlaGatherRow)


def prefill_valid(C, idx, n, pos, skv):
    """`idx >= 0` is NOT its sign bit; `idx < skv` and `pos < len` (signed) the borrows of `(a ^ 2^31) - (b ^ 2^31)`."""
    flip = lambda x: list(x[:31]) + [NOT(x[31])]  # noqa: E731
    return [and_reduce(C, [NOT(idx[31]), sub(C, flip(idx), flip(skv), 32)[1], sub(C, flip(pos), flip(n), 32)[1]])]


FlashMlaPrefillValidV2 = definition("FlashMlaPrefillValid", 2, prefill_valid,
                                    (("idx", I32), ("len", I32), ("pos", I32), ("skv", I32)), BIT,
                                    word=bind(P.FlashMlaPrefillValid), doc="FlashMlaPrefillValid_v1 on bits: " + prefill_valid.__doc__)


def _pair(B, S, q, k, nm_prev=None, l_prev=None, o_prev=None):
    """`FlashMlaPrefillPair_v1`'s body on bits: the score chain `DotBf16_v3{K = D, DOT}`, the P V step `DOT`."""
    return P.pair(bit_ops(B), S, bind(BG.DotBf16, K=S.D, DOT=S.DOT), q, k, nm_prev, l_prev, o_prev)


FlashMlaPrefillPairV2 = flat("FlashMlaPrefillPair", 2, ["D", "BN", "FIRST", "DOT", "SCALE"], lambda S: P._pair_type(S, BIT_T),
                             _pair, P.FlashMlaPrefillPair, lifted=_LIFTED)


def _head(B, S, q, k, sink):
    """`FlashMlaPrefillHead_v1`'s body on bits over `FlashMlaPrefillPair_v2`."""
    def pair_of(first):
        return bind(FlashMlaPrefillPairV2, D=S.D, BN=S.BN, FIRST=first, DOT=S.DOT, SCALE=S.SCALE)
    return P.head(bit_ops(B), S, pair_of, q, k, sink)


FlashMlaPrefillHeadV2 = flat("FlashMlaPrefillHead", 2, ["NP", "D", "BN", "DOT", "SCALE"], lambda S: P._head_type(S, BIT_T), _head,
                             P.FlashMlaPrefillHead, lifted=_LIFTED)


def _prefill(B, S, q, kv, idx, n, skv, sink):
    """`FlashMlaPrefill_v1`'s body on bits over `FlashMlaPrefillValid_v2` and `FlashMlaPrefillHead_v2`."""
    head_of = bind(FlashMlaPrefillHeadV2, NP=S.NP, D=S.D, BN=S.BN, DOT=S.DOT, SCALE=S.SCALE)
    return P.prefill(bit_ops(B), S, FlashMlaPrefillValidV2, head_of, q, kv, idx, n, skv, sink)


FlashMlaPrefillV2 = flat("FlashMlaPrefill", 2, ["NH", "NP", "D", "BN", "DOT", "SCALE"], lambda S: P._prefill_type(S, BIT_T), _prefill,
                         P.FlashMlaPrefill, lifted=_LIFTED)

BOOLEAN_FLASHMLA_PREFILL = [E8m0ShlBf16V2, FlashMlaGatherRowV2, FlashMlaPrefillValidV2, FlashMlaPrefillPairV2, FlashMlaPrefillHeadV2,
                            FlashMlaPrefillV2]
