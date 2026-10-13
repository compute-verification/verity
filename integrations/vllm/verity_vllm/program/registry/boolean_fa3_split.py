"""`fa3_split`'s split-KV dense prefill on bits: version 2 of each Definition, its word body line for line over Boolean Definitions
(the head's body is `fa3_split.head_split`, written once), so each one's word view is its `_v1` at the same statics with the
Boolean ``DOT`` and ``INV`` read as their word views (`boolean_attention._on_bits`).

    AttnBlockFirst_v2{D,NVIS,BN,DOT}                              `boolean_attention`'s block at FIRST, without the running state
    AttentionHeadSplit_v2{T,NBM,NS,D,BN,DOT,INV,MASKED_FROM}      its blocks `AttnBlockFirst_v2` and `AttnBlock_v9`, its split end
                                                                  and combine `SparseMlaSplitEnd_v2` and `SparseMlaCombine_v2`
                                                                  (`boolean_sparse_mla_split`)
    AttentionSplit_v2{T,NBM,NS,NH,KVH,D,BN,DOT,INV,MASKED_FROM}   GQA over `AttentionHeadSplit_v2`

DOT `gemm.HopperBF16WgmmaDot16_v2`, INV `boolean_attention.Fa3InvSum_v2` (`boolean_attention.ATTENTION_FA3`).
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind
from verity_vllm.program.registry import boolean_attention as BA
from verity_vllm.program.registry import boolean_sparse_mla_split as BX
from verity_vllm.program.registry import fa3_split as FS
from verity_vllm.program.registry.boolean_attention import BF16, F32


def _first(B, S, q, k, v):
    """`AttnBlockFirst_v1`'s body on bits: `AttnBlock_v9{D, NVIS, FIRST = true, BN, DOT, CHECK = true}`'s without the running
    state."""
    return BA._block(B, "AttnBlockFirst_v2", S.D, S.NVIS, True, S.BN, S.DOT, q, k, v, None, None, None)


AttnBlockFirstV2 = BA._on_bits("AttnBlockFirst", 2, ["D", "NVIS", "BN", "DOT"], lambda S: FS._first_type(S, BF16, F32), _first,
                               FS.AttnBlockFirst)


def _head_split(B, S, q, k, v):
    """`AttentionHeadSplit_v1`'s body on bits over `AttnBlockFirst_v2`, `AttnBlock_v9`, `SparseMlaSplitEnd_v2` and
    `SparseMlaCombine_v2`."""
    c = BA._Bits(B)
    return FS.head_split(B, S, lambda nvis: bind(AttnBlockFirstV2, D=S.D, NVIS=nvis, BN=S.BN, DOT=S.DOT),
                         lambda b, nvis: bind(BA.AttnBlockV9, D=S.D, NVIS=nvis, FIRST=False, BN=S.BN, DOT=S.DOT,
                                              CHECK=b >= S.MASKED_FROM),
                         bind(BX.SparseMlaSplitEndV2, DV=S.D, INV=S.INV, SCALE=FS.scale(S.D)),
                         lambda n: bind(BX.SparseMlaCombineV2, NSPLIT=n, DV=S.D, INV=S.INV), BF16, lambda: c.word(0, 16), q, k, v)


AttentionHeadSplitV2 = BA._on_bits("AttentionHeadSplit", 2, ["T", "NBM", "NS", "D", "BN", "DOT", "INV", "MASKED_FROM"],
                                   BA._head_type, _head_split, FS.AttentionHeadSplit)


def _split(B, S, q, kc, vc):
    """`AttentionSplit_v1`'s body on bits over `AttentionHeadSplit_v2`."""
    head = bind(AttentionHeadSplitV2, T=S.T, NBM=S.NBM, NS=S.NS, D=S.D, BN=S.BN, DOT=S.DOT, INV=S.INV, MASKED_FROM=S.MASKED_FROM)
    return BA._attention(B, S.T, S.NH, S.KVH, S.D, head, q, kc, vc)


AttentionSplitV2 = BA._on_bits("AttentionSplit", 2, ["T", "NBM", "NS", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM"],
                               BA._attention_type, _split, FS.AttentionSplit)

BOOLEAN_FA3_SPLIT = [AttnBlockFirstV2, AttentionHeadSplitV2, AttentionSplitV2]

__all__ = ["AttentionHeadSplitV2", "AttentionSplitV2", "AttnBlockFirstV2", "BOOLEAN_FA3_SPLIT"]
