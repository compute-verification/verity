"""`fa3_split`'s split dense FA3 prefill on bits: `AttentionHeadSplit_v2` and `AttentionSplit_v2`, `fa3_split`'s bodies (written
once, `fa3_split.head_split` / `attention`) over Boolean Definitions, so each one's word view is its `_v1` at the same statics with
the Boolean ``DOT`` and ``INV`` read as their word views.

    AttentionHeadSplit_v2{T,D,BN,DOT,INV,MASKED_FROM,PER}      blocks `AttnBlock_v9` (`boolean_attention`), each split's end
                                                              `SparseMlaSplitEnd_v2`, the row's `SparseMlaCombine_v2`
    AttentionSplit_v2{T,NH,KVH,D,BN,DOT,INV,MASKED_FROM,PER}    GQA over `AttentionHeadSplit_v2`

DOT is `gemm.HopperBF16WgmmaDot16_v2` and INV `boolean_attention.Fa3InvSum_v2` on FA3 (`boolean_attention.ATTENTION_FA3`).
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import boolean_attention as BAt
from verity.ml.definitions.registry import boolean_sparse_mla_split as BX
from verity.ml.definitions.registry import fa3_split as F

_STATICS = ["T", "D", "BN", "DOT", "INV", "MASKED_FROM", "PER"]


def _head(B, S, q, k, v):
    """`AttentionHeadSplit_v1`'s body on bits over `AttnBlock_v9`, `SparseMlaSplitEnd_v2` and `SparseMlaCombine_v2`."""
    c = BAt._Bits(B)

    def block_of(nvis, first, check):
        return bind(BAt.AttnBlockV9, D=S.D, NVIS=nvis, FIRST=first, BN=S.BN, DOT=S.DOT, CHECK=check)
    end = bind(BX.SparseMlaSplitEndV2, DV=S.D, INV=S.INV, SCALE=S.D ** -0.5)
    return F.head_split(B, S, block_of, end, lambda n: bind(BX.SparseMlaCombineV2, NSPLIT=n, DV=S.D, INV=S.INV),
                        lambda w: c.word(0, w), q, k, v)


AttentionHeadSplitV2 = BAt._on_bits("AttentionHeadSplit", 2, _STATICS, lambda S: F._head_type(S, BAt.BF16), _head, F.AttentionHeadSplit)


def _attention(B, S, q, kc, vc):
    """`AttentionSplit_v1`'s body on bits over `AttentionHeadSplit_v2`; every head of the row shares its tile."""
    head = bind(AttentionHeadSplitV2, T=S.T, D=S.D, BN=S.BN, DOT=S.DOT, INV=S.INV, MASKED_FROM=S.MASKED_FROM, PER=S.PER)
    return F.attention(B, S, head, q, kc, vc, 16)


AttentionSplitV2 = BAt._on_bits("AttentionSplit", 2, ["T", "NH", "KVH"] + _STATICS[1:], lambda S: F._attention_type(S, BAt.BF16),
                                _attention, F.AttentionSplit)

BOOLEAN_FA3_SPLIT = [AttentionHeadSplitV2, AttentionSplitV2]

__all__ = ["AttentionHeadSplitV2", "AttentionSplitV2", "BOOLEAN_FA3_SPLIT"]
