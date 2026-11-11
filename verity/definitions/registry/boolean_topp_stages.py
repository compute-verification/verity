"""Top-p's stage units on bits: `topp_stages`' five families as And/Xor/Not circuits, each σ of its word unit.

    TopPStatsHead_v2{N,FIRST}             word view: TopPStatsHead_v1 at the same statics
    TopPStatsExp_v2{N,SPT,AT,LAST,FIRST}  word view: TopPStatsExp_v1
    TopPProbs_v2{V,S,N,HEAD}              word view: TopPProbs_v1
    TopPStep_v2{V,S,N,FIRST,COMBINE}      word view: TopPStep_v1
    GumbelTopPSelect_v2{V,N,FIRST,LAST}   word view: GumbelTopPSelect_v1

A word unit's body is primitive Calls and batches of them (`dead_free.live_replay`), so its σ (`boolean.sigma_lift`) is the same
Calls over the bits, each of the Boolean version of its primitive. The MUFU operations among them are their gates
(`MufuEx2Ftz_v2`, `DivFullRcp_v2`; `verity.definitions.ops.boolean.mufu`), where the word unit reads a pinned table: a unit on
bits reads none, so a statement holds it with no lookup slot and its unit returns its own outputs.

Four of `topp_word_gates`' primitives get their Boolean versions here (`BitXor_v2`, `U32Xor_v2`, `SelectBit_v2`,
`TopPNumkeep_v2`); the libdevice log's and the Gumbel lane's are `gumbel_boolean`'s (`NvLogfQuietNan_v1`,
`GumbelNoiseLane_v3`).
"""
from __future__ import annotations

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT, ONE, OR, ZERO, and_reduce, const_bits, mux, or_reduce, sub
from verity.circuits.boolean.trace import definition
from verity.definitions.silicon import fp
from verity.circuits.types import Array
from verity.definitions.boolean import sigma_lift
from verity.definitions.registry import gumbel_boolean  # noqa: F401  (the log's and the lane's Boolean versions, which σ reads)
from verity.definitions.registry import topp_stages as T
from verity.definitions.registry import topp_word_gates as TG

__all__ = ["BOOLEAN_STAGE_DEFINITIONS", "BitXor", "GumbelTopPSelect", "SelectBit", "TopPNumkeep", "TopPProbs", "TopPStatsExp",
           "TopPStatsHead", "TopPStep", "U32Xor", "topp_numkeep"]

BIT = BL.BIT
U32 = Array(32, BIT)


def topp_numkeep(C, nd, q):
    """`TopPNumkeep_v1`: ``min(max(nd - qi, 1), nd)`` on u32 counts, qi the f32 ``q`` truncated to a u32 (0 unless q is finite
    and above zero, all ones from 2^32 on); 0 when ``nd`` is 0."""
    s, e, m = q[31], list(q[23:31]), list(q[:23])
    finite = C.AND(NOT(s), NOT(and_reduce(C, e)))
    below_one = sub(C, e, const_bits(127, 8), 8)[1]
    past_u32 = NOT(sub(C, e, const_bits(127 + 32, 8), 8)[1])
    whole = fp.shl(C, m + [ONE] + [ZERO] * 32, sub(C, e, const_bits(127, 8), 8)[0][:5])[23:55]
    qi = [C.AND(finite, OR(C, past_u32, C.AND(NOT(below_one), b))) for b in whole]
    keep = mux(C, sub(C, qi, nd, 32)[1], sub(C, nd, qi, 32)[0], const_bits(1, 32))
    some = or_reduce(C, nd)
    return [C.AND(some, b) for b in keep]


BitXor = definition("BitXor", 2, lambda C, a, b: [a[0] ^ b[0]], (("a", BIT), ("b", BIT)), BIT, word=TG.BitXor,
                    doc="a XOR b on one bit: BitXor_v1")
U32Xor = definition("U32Xor", 2, lambda C, a, b: [x ^ y for x, y in zip(a, b)], (("a", U32), ("b", U32)), U32, word=TG.U32Xor,
                    doc="a XOR b on 32-bit words' bits: U32Xor_v1")
SelectBit = definition("SelectBit", 2, lambda C, c, a, b: mux(C, c[0], a, b), (("c", BIT), ("a", BIT), ("b", BIT)), BIT,
                       word=TG.SelectBit, doc="c ? a : b on one bit: SelectBit_v1")
TopPNumkeep = definition("TopPNumkeep", 2, topp_numkeep, (("numdup", U32), ("q", U32)), U32, word=TG.TopPNumkeep,
                         doc="The combine's numkeep on bits: TopPNumkeep_v1")

TopPStatsHead = sigma_lift(T.TopPStatsHead, 2)
TopPStatsExp = sigma_lift(T.TopPStatsExp, 2)
TopPProbs = sigma_lift(T.TopPProbs, 2)
TopPStep = sigma_lift(T.TopPStep, 2)
GumbelTopPSelect = sigma_lift(T.GumbelTopPSelect, 2)

#: each stage family on bits, in `topp_stages.STAGE_DEFINITIONS`' order
BOOLEAN_STAGE_DEFINITIONS = [TopPStatsHead, TopPStatsExp, TopPProbs, TopPStep, GumbelTopPSelect]
