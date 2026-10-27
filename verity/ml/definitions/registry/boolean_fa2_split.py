"""`fa2_split`'s split-KV decode on bits: version 2 of each Definition, its word body line for line over Boolean Definitions (the
bodies are `fa2_split`'s, written once over `sparse_mla.Steps`), so each one's word view is its `_v1` at the same statics with the
Boolean ``DOT`` and ``INV`` read as their word views.

    Fa2SplitEnd_v2{D,INV,SCALE}                                 INV: `ops.boolean.attention.Fa2InvSum_v2`
    Fa2Combine_v2{NS,D}
    Fa2AttentionHeadSplit_v2{T,NS,D,BN,DOT,INV,MASKED_FROM}     DOT: `gemm.HopperBF16WgmmaDot16_v2` on sm_120; the blocks and the
    Fa2AttentionSplit_v2{T,NS,NH,KVH,D,BN,DOT,INV,MASKED_FROM}  NS = 1 head are `boolean_attention`'s `AttnBlock_v8` / `AttentionHead_v8`

The callees are `boolean_sparse_mla_split.BITS` (MUFU.LG2, the FTZ comparisons and `SelectI32_v2` among them).  The end and the
combine are each one traced circuit (`boolean_dsa.flat`); the heads are composites of their blocks', ends' and combine's Calls,
as `AttentionHead_v8` is.
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import attention as BA
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.refs import Coll, concat, strided_view
from verity.ml.circuits.types import Array
from verity.ml.definitions.registry import boolean_attention as BAT
from verity.ml.definitions.registry import boolean_sparse_mla_split as BSS
from verity.ml.definitions.registry import fa2_split as W
from verity.ml.definitions.registry.boolean_attention import BF16, F32
from verity.ml.definitions.registry.boolean_dsa import flat

#: FA2's DOT and INV on sm_120 on bits: its mma.sync m16n8k16 is Hopper's BF16 pipeline there (`sm120.mma.m16n8k16.bf16`)
ATTENTION_FA2_SM120: dict = {"DOT": BG.HopperBF16WgmmaDot16, "INV": BA.Fa2InvSum}


def _end(B, S, m, l, o):
    """`Fa2SplitEnd_v1`'s body on bits."""
    return W.split_end(BSS.bit_steps(B), S, m, l, o)


Fa2SplitEndV2 = flat("Fa2SplitEnd", 2, ["D", "INV", "SCALE"], lambda S: W._end_type(S, F32), _end, W.Fa2SplitEnd, lifted=("INV",))


def _combine(B, S, lse, o):
    """`Fa2Combine_v1`'s body on bits."""
    return W.combine(BSS.bit_steps(B), S, lse, o)


Fa2CombineV2 = flat("Fa2Combine", 2, ["NS", "D"], lambda S: W._combine_type(S, BF16, F32), _combine, W.Fa2Combine)


def _head_split(B, S, q, k, v):
    """`Fa2AttentionHeadSplit_v1`'s body on bits over `AttnBlock_v8`, `AttentionHead_v8` (NS = 1), `Fa2SplitEnd_v2` and
    `Fa2Combine_v2`."""
    def block_of(nvis, first, check):
        return bind(BAT.AttnBlockV8, D=S.D, NVIS=nvis, FIRST=first, BN=S.BN, DOT=S.DOT, CHECK=check)
    unsplit = bind(BAT.AttentionHeadV8, T=S.T, D=S.D, BN=S.BN, DOT=S.DOT, INV=S.INV, MASKED_FROM=S.MASKED_FROM)
    end = bind(Fa2SplitEndV2, D=S.D, INV=S.INV, SCALE=S.D ** -0.5)
    return W.head_split(BSS.bit_steps(B), S, block_of, unsplit, end, bind(Fa2CombineV2, NS=S.NS, D=S.D), q, k, v)


_HEAD = ["T", "NS", "D", "BN", "DOT", "INV", "MASKED_FROM"]
Fa2AttentionHeadSplitV2 = BAT._on_bits("Fa2AttentionHeadSplit", 2, _HEAD, lambda S: W._head_type(S, BF16), _head_split,
                                       W.Fa2AttentionHeadSplit)


def _split(B, S, q, kc, vc):
    """`Fa2AttentionSplit_v1`'s body on bits over `Fa2AttentionHeadSplit_v2` (a bf16 word is 16 bit elements: KV head g's rows
    are 16 D bits at stride 16 KVH D)."""
    T, NH, KVH, D = S.T, S.NH, S.KVH, S.D
    G, Wd = NH // KVH, 16 * D
    head = bind(Fa2AttentionHeadSplitV2, **{k: getattr(S, k) for k in _HEAD})
    outs = []
    for g in range(KVH):
        kg = Coll(Array(T, Array(D, BF16)), strided_view(kc, g * Wd, Wd, KVH * Wd, 1, T * Wd))
        vg = Coll(Array(T, Array(D, BF16)), strided_view(vc, g * Wd, Wd, KVH * Wd, 1, T * Wd))
        outs.append(B.batch(head, q[g * G:(g + 1) * G], kg, vg, axes=(0, None, None)))
    return Coll(Array(NH, Array(D, BF16)), concat([o.refs for o in outs]))


Fa2AttentionSplitV2 = BAT._on_bits("Fa2AttentionSplit", 2, ["T", "NS", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM"],
                                   lambda S: W._split_type(S, BF16), _split, W.Fa2AttentionSplit)

BOOLEAN_FA2_SPLIT = [Fa2SplitEndV2, Fa2CombineV2, Fa2AttentionHeadSplitV2, Fa2AttentionSplitV2]
