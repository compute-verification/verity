"""`sparse_mla`'s FA3 sparse MLA attention on bits: version 2 of each Definition, its word body line for line over Boolean
Definitions (the bodies are `sparse_mla`'s, written once over `sparse_mla.Steps`), so each one's word view is its `_v1` at the
same statics with the Boolean ``DOT`` and ``INV`` read as their word views.

    SparseMlaBlock_v2{DR,DV,NVIS,FIRST,BN,DOT,SCALE}   DOT: `gemm.HopperBF16WgmmaDot16_v2` (FA3's wgmma k16 step)
    SparseMlaHead_v2{T,DR,DV,BN,DOT,INV,SCALE}         INV: `boolean_attention.Fa3InvSum_v2`
    SparseMlaHeadBlock_v2{DR,DV,NVIS,FIRST,BN,DOT,SCALE}
    SparseMlaHeadBlock_v3{DR,DV,NVIS,FIRST,BN,DOT,SCALE}   the same function written as Calls; ``opt_in``
    SparseMlaHeadScores_v2{DR,DV,NVIS,FIRST,BN,DOT,SCALE}  the block up to P, written as Calls
    SparseMlaHeadValues_v2{NVIS,FIRST,BN,DOT,G}            the block's P V steps on G columns of O, written as Calls
    SparseMlaHeadEnd_v2{DV,INV}
    SparseMla_v2{T,NH,DR,DV,BN,DOT,INV,SCALE}

The callees are `boolean_attention`'s: `DotBf16_v3{K = DR + DV, DOT}` and `DOT`, `F2fpBf16_v2`, the FTZ f32 operations,
`F32Max_v2`, `GuardNegInfZero_v2`, `MufuEx2Ftz_v2` and the inverse sum; a constant word is constant bits.  No table or lookup gate.
Each `_v2` is one traced circuit (`boolean_dsa.flat`), so the query every score reads and the rescale every output reads are
decoded once; its description is every gate of it.

`SparseMlaHeadBlock_v3` is the key block whose description does not grow with its gates: `boolean_attention`'s block form over
this block's operands, a batch of one Definition per row of k16 steps.  A full step is cut at both operands, so no operand of
one is decoded twice: the 16 words of the query (the same for the block's NVIS keys) and of bf16(P) (the same for the DV
columns) are decoded once for their row (`AttnDotStepA`), and each word of a key row once for the block
(`SparseMlaDotStepB_v1{DOT}`), whether its score's step reads it, its P V step or both, as a kv_c word is by both.
`SparseMlaDotStep_v1{DOT}` is the rest of the step, from the two frontiers.  The online softmax between the chains is the
attention blocks' (`boolean_attention._softmax`: the scale word folded into the gates that read it, the rescale decoded once a
row).  Three things the block computes again, which one trace of it doesn't (circuit-check counts them as redundant gates, a
warning): a chain starts from the zero accumulator's bits, and each member of its first step computes the step's gates on
those constants; the first visited block's partial step (NVIS not a multiple of 16) is an `AttnDotStep{DOT,NV,FROM_ZERO}`, P's
zero words folded in, on its keys' words and zero rows, so it decodes those keys' kv_c words a second time; and the softmax's
per-key callees repeat what they repeat in the attention blocks.  It is ``opt_in`` (`verity.definitions.boolean`):
`boolean_version` keeps `_v2`, and `_v3` serves only a caller that binds it.

`SparseMlaHeadScores_v2` and `SparseMlaHeadValues_v2` are `SparseMlaHeadBlock_v3` cut at P, over the same Calls: the scores
unit is its key words' own gates, its score rows and its softmax; a values unit is the rescale of its G columns of O and their
P V rows, on those columns of the keys' kv_c, the only key words it takes, whose own gates it computes (the block computes a
kv_c word's once for its score and its P V step, so the units together hold those gates once more than the block), and only
for the keys of a whole step.
"""
from __future__ import annotations

from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.definitions.ops.boolean import attention as BA
from verity.definitions.ops.boolean import elementwise as EW
from verity.definitions.ops.boolean import gemm as BG
from verity.definitions.ops.boolean import mufu as BM
from verity.circuits.refs import Coll, array_of, concat, strided_view, tuple_of
from verity.circuits.types import Array
from verity.definitions.registry import sparse_mla as M
from verity.definitions.registry.boolean_attention import (BF16, F32, AttnDotStep, AttnDotStepA, Fa3InvSum, _Bits, _scale_row,
                                                            _softmax, _step_row)
from verity.definitions.registry.boolean_dsa import flat, replay

BITS = dict(max=BA.F32Max, guard=BA.GuardNegInfZero, ex2=BM.MufuEx2Ftz, mul=BA.F32MulFtz, sub=BA.F32SubFtz, fmasub=BA.F32FmaSubFtz,
            fma=BA.F32FmaFtz, add=BA.F32AddFtz, cvt=BG.F2fpBf16)

#: GLM-5.3 at TP8 on bits: `sparse_mla.SPARSE_MLA_GLM53` with the Boolean DOT and INV
SPARSE_MLA_GLM53 = {**M.SPARSE_MLA_GLM53, "DOT": BG.HopperBF16WgmmaDot16, "INV": Fa3InvSum}
_LIFTED = ("DOT", "INV")


def bit_steps(B) -> M.Steps:
    c = _Bits(B)
    return M.Steps(B, BITS, lambda bits: c.word(bits, 32), lambda: c.word(0, 16), 16, BF16, F32)


def _block(B, S, q, kv, m_prev=None, l_prev=None, o_prev=None):
    """`SparseMlaBlock_v1`'s body on bits: the S chain `DotBf16_v3{K = DR + DV, DOT}`, the P V step `DOT`."""
    M.check_block_statics(S.DR, S.DV, S.NVIS, S.BN)
    return M.block(bit_steps(B), S, bind(BG.DotBf16, K=S.DR + S.DV, DOT=S.DOT), S.DOT, q, kv, m_prev, l_prev, o_prev)


SparseMlaBlockV2 = flat("SparseMlaBlock", 2, ["DR", "DV", "NVIS", "FIRST", "BN", "DOT", "SCALE"],
                        lambda S: M._block_type(S, BF16, F32), _block, M.SparseMlaBlock, lifted=_LIFTED)


def _head(B, S, q_nope, q_rope, kv):
    """`SparseMlaHead_v1`'s body on bits over `SparseMlaBlock_v2`, epilogue `INV`."""
    def tile_of(nvis, first):
        return bind(SparseMlaBlockV2, DR=S.DR, DV=S.DV, NVIS=nvis, FIRST=first, BN=S.BN, DOT=S.DOT, SCALE=S.SCALE)
    return M.head(bit_steps(B), S, tile_of, S.INV, q_nope, q_rope, kv)


SparseMlaHeadV2 = flat("SparseMlaHead", 2, ["T", "DR", "DV", "BN", "DOT", "INV", "SCALE"], lambda S: M._head_type(S, BF16), _head,
                       M.SparseMlaHead, lifted=_LIFTED)


def _head_block(B, S, q_nope, q_rope, kv, m_prev=None, l_prev=None, o_prev=None):
    """`SparseMlaHeadBlock_v1`'s body on bits: the S chain `DotBf16_v3{K = DR + DV, DOT}`, the P V step `DOT`."""
    return M.head_block(bit_steps(B), S, bind(BG.DotBf16, K=S.DR + S.DV, DOT=S.DOT), S.DOT, q_nope, q_rope, kv, m_prev, l_prev,
                        o_prev)


SparseMlaHeadBlockV2 = flat("SparseMlaHeadBlock", 2, ["DR", "DV", "NVIS", "FIRST", "BN", "DOT", "SCALE"],
                            lambda S: M._head_block_type(S, BF16, F32), _head_block, M.SparseMlaHeadBlock, lifted=_LIFTED)


def _full_step_type(S):
    return bind(AttnDotStep, DOT=S.DOT, NV=16, FROM_ZERO=False).signature()


def _full_step(S):
    """`AttnDotStep{DOT, NV = 16, FROM_ZERO = False}`'s circuit over its operands' bits, a's own gates among them."""
    c = BS.compile(bind(AttnDotStep, DOT=S.DOT, NV=16, FROM_ZERO=False))
    return lambda C, *elements: replay(c, C, [f for e in elements for f in e])


SparseMlaDotStepB, SparseMlaDotStep = EW.shared_specialized(
    ("SparseMlaDotStepB", "SparseMlaDotStep"), 1, ("DOT",), _full_step, _full_step_type, 1,
    ("the gates of DOT(acc, a, b) that read only one bf16 word of b, once per word whichever steps read it",
     "DOT(acc, a, b) on bits from a's own gates ad (AttnDotStepA{DOT,NV=16,FROM_ZERO=False}) and each word of b's own gates bd"),
    by_element=True)


def _own_gates(part):
    """A builder for ``part{DOT}`` of the step, its own gates on a circuit: with one, Flock's flat lowering runs each activation
    of the part (`ir_lower.lower_gates`) and keeps a word for what it returns, not for each of its gates. The circuit it
    lowers to is the same."""
    return lambda C, S, *elements: replay(BS.compile(bind(part, DOT=S.DOT)), C, [f for e in elements for f in e])


SparseMlaDotStepB.build = _own_gates(SparseMlaDotStepB)  # type: ignore[attr-defined]
SparseMlaDotStep.build = _own_gates(SparseMlaDotStep)  # type: ignore[attr-defined]


def _head_block_calls(B, S, q_nope, q_rope, kv, m_prev=None, l_prev=None, o_prev=None):
    """`SparseMlaHeadBlock_v1`'s body on bits as Calls: S[c] and P V by rows of `SparseMlaDotStep` (the query's 16 words of a
    step decoded once for the NVIS keys, bf16(P)'s once for the DV columns, each key word once for the block), a partial P V
    step by a row of `AttnDotStep`, and the online softmax `boolean_attention`'s, guarded only in the first visited block."""
    DR, DV, NVIS, FIRST, BN, DOT = S.DR, S.DV, S.NVIS, S.FIRST, S.BN, S.DOT
    M.check_block_statics(DR, DV, NVIS, BN)
    if NVIS != BN and not FIRST:
        raise ValueError(f"SparseMlaHeadBlock: NVIS={NVIS} keys of BN={BN}, and only the first visited block is partial")
    c, KW = _Bits(B), DR + DV
    own_a, step = bind(AttnDotStepA, DOT=DOT, NV=16, FROM_ZERO=False), bind(SparseMlaDotStep, DOT=DOT)
    kd = B.batch(bind(SparseMlaDotStepB, DOT=DOT), Coll(Array(NVIS * KW, BF16), kv.refs), axes=(0,))
    D = kd.type.elem

    def words(rows: int, per_row: int, at: int) -> tuple[Coll, Coll]:
        """Words [at, at + per_row) of ``rows`` key rows from row at // KW on, and their own gates."""
        return (Coll(Array(rows, Array(per_row, BF16)), strided_view(kv, 16 * at, 16 * per_row, 16 * KW, 1, 16 * per_row * rows)),
                Coll(Array(rows, Array(per_row, D)), strided_view(kd, D.leaves * at, D.leaves * per_row, D.leaves * KW, 1,
                                                                 D.leaves * per_row * rows)))

    def row(a: Coll, b: Coll, bd: Coll, acc: Coll | None, n: int, axis: int) -> Coll:
        """One step per member of a row of ``n`` that shares the 16 words ``a``: member j's words of ``b`` along ``axis``."""
        acc = c.zeros(Array(n, F32)) if acc is None else acc
        return B.batch(step, a, b, acc, B.call(own_a, a), bd, axes=(None, axis, 0, None, axis))

    q = Coll(Array(KW, BF16), concat([q_rope.refs, q_nope.refs]))
    scores = None
    for t in range(KW // 16):
        # a key row is [kv_c | k_pe], the chain reads [k_pe | kv_c]
        scores = row(q[16 * t:16 * t + 16], *words(NVIS, 16, DV + 16 * t if 16 * t < DR else 16 * t - DR), scores, NVIS, 0)
    m_new, lanes, pb, scores_scale = _softmax(B, c, NVIS, FIRST, scores, m_prev, l_prev, FIRST, FIRST, None,
                                              M.scale_log2_bits(S.SCALE))
    o = None if FIRST else _scale_row(B, o_prev, scores_scale)
    for k0 in range(0, NVIS, 16):
        nv = min(16, NVIS - k0)
        prow = Coll(Array(nv, BF16), concat([x.refs for x in pb[k0:k0 + nv]]))
        v, vd = words(nv, DV, k0 * KW)
        if nv == 16:
            o = row(prow, v, vd, o, DV, 1)
        else:
            # the rows FA3 loads past the block's keys are zero (`sparse_mla._padded`), and the step reads a pad word of V
            v = Coll(Array(16, Array(DV, BF16)), concat([v.refs, c.zeros(Array(16 - nv, Array(DV, BF16))).refs]))
            o = _step_row(B, DOT, nv, prow, v, o, axis=1)
    return tuple_of(m_new, array_of(lanes), o)


SparseMlaHeadBlockV3 = EW.on_bits("SparseMlaHeadBlock", 3, ["DR", "DV", "NVIS", "FIRST", "BN", "DOT", "SCALE"],
                                  lambda S: M._head_block_type(S, BF16, F32), _head_block_calls, M.SparseMlaHeadBlock,
                                  lifted=_LIFTED)
SparseMlaHeadBlockV3.opt_in = True  # type: ignore[attr-defined]


def _words(c: Coll, elem, rows: int, n: int, at: int, stride: int) -> Coll:
    """Elements [at, at + n) of each of ``rows`` rows of ``stride`` elements of ``c``, a row of ``n`` each."""
    L = elem.leaves
    return Coll(Array(rows, Array(n, elem)), strided_view(c, L * at, L * n, L * stride, 1, L * n * rows))


def _dot_row(B, c: _Bits, DOT, a: Coll, b: Coll, bd: Coll, acc: Coll | None, n: int, axis: int) -> Coll:
    """One `SparseMlaDotStep` per member of a row of ``n`` that shares the 16 words ``a``, their own gates computed once: member
    j's words of ``b`` and those words' own gates ``bd`` along ``axis``, from ``acc`` (None: from +0)."""
    acc = c.zeros(Array(n, F32)) if acc is None else acc
    own_a = bind(AttnDotStepA, DOT=DOT, NV=16, FROM_ZERO=False)
    return B.batch(bind(SparseMlaDotStep, DOT=DOT), a, b, acc, B.call(own_a, a), bd, axes=(None, axis, 0, None, axis))


def _head_scores_calls(B, S, q_nope, q_rope, kv, m_prev=None, l_prev=None):
    """`SparseMlaHeadScores_v1`'s body on bits as Calls, `SparseMlaHeadBlock_v3`'s up to P: each key word's own gates once
    (`SparseMlaDotStepB`), S[c] by rows of `SparseMlaDotStep` that share the query's 16 words of a step, and the online softmax
    `boolean_attention`'s, guarded only in the first visited block."""
    DR, DV, NVIS, FIRST, DOT = S.DR, S.DV, S.NVIS, S.FIRST, S.DOT
    M.check_block_statics(DR, DV, NVIS, S.BN)
    M._whole_or_first("SparseMlaHeadScores", S)
    c, KW = _Bits(B), DR + DV
    kd = B.batch(bind(SparseMlaDotStepB, DOT=DOT), Coll(Array(NVIS * KW, BF16), kv.refs), axes=(0,))
    q = Coll(Array(KW, BF16), concat([q_rope.refs, q_nope.refs]))
    scores = None
    for t in range(KW // 16):
        at = DV + 16 * t if 16 * t < DR else 16 * t - DR  # a key row is [kv_c | k_pe], the chain reads [k_pe | kv_c]
        scores = _dot_row(B, c, DOT, q[16 * t:16 * t + 16], _words(kv, BF16, NVIS, 16, at, KW),
                          _words(kd, kd.type.elem, NVIS, 16, at, KW), scores, NVIS, 0)
    m_new, lanes, pb, rescale = _softmax(B, c, NVIS, FIRST, scores, m_prev, l_prev, FIRST, FIRST, None,
                                         M.scale_log2_bits(S.SCALE))
    return tuple_of(m_new, array_of(lanes), array_of(pb), *(() if FIRST else (rescale,)))


SparseMlaHeadScoresV2 = EW.on_bits("SparseMlaHeadScores", 2, ["DR", "DV", "NVIS", "FIRST", "BN", "DOT", "SCALE"],
                                   lambda S: M._head_scores_type(S, BF16, F32), _head_scores_calls, M.SparseMlaHeadScores,
                                   lifted=_LIFTED)


def _head_values_calls(B, S, pb, kv, rescale=None, o_prev=None):
    """`SparseMlaHeadValues_v1`'s body on bits as Calls, `SparseMlaHeadBlock_v3`'s P V steps on G columns of O: each column of
    the O before by the rescale (its own gates once), then rows of `SparseMlaDotStep` that share bf16(P)'s 16 words of a step,
    on the keys' words of those columns and their own gates (`SparseMlaDotStepB`, once a word); a partial step by a row of
    `AttnDotStep` over its keys' words and zero rows."""
    NVIS, FIRST, DOT, G = S.NVIS, S.FIRST, S.DOT, S.G
    M.check_values_statics(S)
    c, whole = _Bits(B), NVIS - NVIS % 16
    if whole:
        kd = B.batch(bind(SparseMlaDotStepB, DOT=DOT), Coll(Array(whole * G, BF16), _words(kv, BF16, whole, G, 0, G).refs),
                     axes=(0,))
    o = None if FIRST else _scale_row(B, o_prev, rescale)
    for k0 in range(0, NVIS, 16):
        nv = min(16, NVIS - k0)
        v = _words(kv, BF16, nv, G, k0 * G, G)
        if nv == 16:
            o = _dot_row(B, c, DOT, pb[k0:k0 + 16], v, _words(kd, kd.type.elem, 16, G, k0 * G, G), o, G, 1)
        else:
            # the rows FA3 loads past the block's keys are zero (`sparse_mla._padded`), and the step reads a pad word of V
            v = Coll(Array(16, Array(G, BF16)), concat([v.refs, c.zeros(Array(16 - nv, Array(G, BF16))).refs]))
            o = _step_row(B, DOT, nv, pb[k0:k0 + nv], v, o, axis=1)
    return o


SparseMlaHeadValuesV2 = EW.on_bits("SparseMlaHeadValues", 2, ["NVIS", "FIRST", "BN", "DOT", "G"],
                                   lambda S: M._head_values_type(S, BF16, F32), _head_values_calls, M.SparseMlaHeadValues,
                                   lifted=_LIFTED)


def _head_end(B, S, l, o):
    """`SparseMlaHeadEnd_v1`'s body on bits, epilogue `INV`."""
    return M.head_end(bit_steps(B), S.INV, l, o)


SparseMlaHeadEndV2 = flat("SparseMlaHeadEnd", 2, ["DV", "INV"], lambda S: M._head_end_type(S, BF16, F32), _head_end,
                          M.SparseMlaHeadEnd, lifted=_LIFTED)


def _mla(B, S, q_nope, q_rope, kv):
    """`SparseMla_v1`'s body on bits over `SparseMlaHead_v2`."""
    hd = bind(SparseMlaHeadV2, T=S.T, DR=S.DR, DV=S.DV, BN=S.BN, DOT=S.DOT, INV=S.INV, SCALE=S.SCALE)
    return B.batch(hd, q_nope, q_rope, kv, axes=(0, 0, None))


SparseMlaV2 = flat("SparseMla", 2, ["T", "NH", "DR", "DV", "BN", "DOT", "INV", "SCALE"], lambda S: M._mla_type(S, BF16), _mla,
                   M.SparseMla, lifted=_LIFTED)

BOOLEAN_SPARSE_MLA = [SparseMlaBlockV2, SparseMlaHeadV2, SparseMlaHeadBlockV2, SparseMlaDotStepB, SparseMlaDotStep,
                      SparseMlaHeadBlockV3, SparseMlaHeadScoresV2, SparseMlaHeadValuesV2, SparseMlaHeadEndV2, SparseMlaV2]
