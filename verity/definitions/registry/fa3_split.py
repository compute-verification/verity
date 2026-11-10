"""FA3's dense causal prefill with the key range split: `AttentionHeadSplit_v1` / `AttentionSplit_v1`, the `Attention_v4` chain
(`fa3_check_inf`) run per split, each split ended as `Softmax::finalize` ends it under `Split` and the splits merged by
`FlashAttnFwdCombine`.

vllm-flash-attn 506341a1 splits a varlen launch when `fa3_model.split_plan`'s static bound exceeds 1 (GLM-5.3 TP8's head-256
prefill at 8 heads: 321 to 1664 rows), each sequence at its own dynamic count NS.  The tile of query rows m (`kBlockM` rows,
`fa3_model.kblock_m_for`) visits key blocks 0 .. NBM - 1, NBM = min(ceil(T_seq / BN), ceil((m + 1) kBlockM / BN))
(`block.h get_n_block_min_max`), and split s of the tile the blocks [s PER, min((s + 1) PER, NBM)), PER = ceil(NBM / NS)
(`split_per`).  A row of T keys sees the blocks below ceil(T / BN): its split s is [s PER, min((s + 1) PER, ceil(T / BN))),
non-empty while s PER < ceil(T / BN) (`row_splits`).  The tile's other splits are fully masked for the row, so their running sum
is 0 and `finalize` writes their lse as -inf, which is the word the combine pads its warp with: `SparseMlaCombine`'s NSPLIT counts
the row's non-empty splits only.  The blocks of a split a row does not see, visited first, leave the zero state (as in
`Attention_v4`, `fa3_model.kernel_geometry`), so the row's first visible block of a split is its `FIRST`.

A split's iterations (`mainloop_fwd_sm90_tma_gmma_ws.hpp`, IntraWGOverlap): its last block first, `Is_first` with `Check_inf`; then
blocks at or above the tile's `n_block_min_causal_local_mask` (MASKED_FROM, `attention_vocab.fa3_masked_from`) with `Check_inf`,
the rest without.  So block b runs `AttnBlock_v4{..., FIRST = (b is the split's first), CHECK = FIRST or b >= MASKED_FROM}`.
Each split ends with `SparseMlaSplitEnd_v1{DV=D, INV, SCALE=D^-0.5}` (the partial o * INV(sum) in f32 and the lse; its
`scale_log2` word is `b1.scale_log2_bits(D)`'s) and the row with `SparseMlaCombine_v1{NSPLIT, DV=D, INV}`.

`SparseMlaCombine` is the combine's Definition at any head: `run_mha_fwd_combine` launches `FlashAttnFwdCombine<Shape<8, 128>,
kLogMaxSplits 5, 256, 1, false, Varlen, bf16, float, Sm90>` for head_v > 64 and at most 32 splits, the instantiation sparse MLA's
decode runs (`sparse_mla_split`, whose docstring reads its SASS); the head dim is a runtime argument.  A launch whose static bound
exceeds 32 runs another instantiation (kLogMaxSplits 6 and up), which `check_combine_bound` refuses.

A sequence whose dynamic count is 1 is written unsplit (`is_split = num_splits > 1` in the varlen epilogue): `Attention_v4`.

With the lse.  FA3 writes every row's lse beside its output, and vLLM reads it when a dense prefill chunk has context (the
context's and the suffix's, which `MergeAttnStates_v1` merges).  The unsplit kernel's `finalize` is the split one's
(hopper/softmax.h:137-152): in the SASS of the head-256 bf16 instances (128x80x256, varlen, causal or not, split or not), its
plain path computes `lse = sum != 0 ? FFMA.FTZ(scale_log2 * LN2, row_max, FMUL.FTZ(MUFU.LG2(sum), LN2)) : -inf` from the
unguarded running max.  Their other path reads attention sinks, which these Definitions don't take.  So
`AttentionHeadLse_v1` is `AttentionHead_v4`'s blocks ended by `SparseMlaSplitEnd_v1` with its partial in bf16, which is
`AttentionHead_v4`'s output word.  `AttentionHeadSplitLse_v1` is `AttentionHeadSplit_v1` with `SparseMlaCombineLse_v1`, the
lse the combine kernel stores.

Registered functions (all `_v1`):
  AttentionHeadSplit{T,D,BN,DOT,INV,MASKED_FROM,PER}   (q: D bf16, k: T x D bf16, v: T x D bf16) -> D bf16
  AttentionSplit{T,NH,KVH,D,BN,DOT,INV,MASKED_FROM,PER} (q: NH x D, kc: T x KVH x D, vc: T x KVH x D) -> NH x D bf16
  AttentionHeadLse{T,D,BN,DOT,INV,MASKED_FROM}         (q: D bf16, k: T x D bf16, v: T x D bf16) -> (D bf16, lse)
  AttentionHeadSplitLse{T,D,BN,DOT,INV,MASKED_FROM,PER} (q: D bf16, k: T x D bf16, v: T x D bf16) -> (D bf16, lse)
"""
from __future__ import annotations

from verity.circuits.defs import bind, composite
from verity.definitions.ops.prims import ZERO32
from verity.circuits.refs import Coll, array_of, concat, strided_view, tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import prims as P
from verity.definitions.registry import sparse_mla_split as W
from verity.definitions.registry.b1 import ZERO16
from verity.definitions.registry.fa3_check_inf import AttnBlockV4
from verity.definitions.registry.prims import BF16, F32

_CONF = ("model-untested: read from vllm-flash-attn 506341a1's split mainloop, epilogue and combine (the module docstring's "
         "files); tests/properties/glm53_fa3_prefill_capture_gpu.py's served-split class is its hardware check")
_LSE_CONF = ("model-untested: the output as AttentionHead_v4 and AttentionHeadSplit_v1 state it, the lse read from "
             "vllm-flash-attn 506341a1's finalize and combine and their cc 9.0 SASS (the module docstring); no capture records "
             "that lse yet, and the check that would establish it is tests/properties/glm53_fa3_prefill_capture_gpu.py with "
             "return_softmax_lse on the context and suffix launches")


def split_per(T_seq: int, row: int, BM: int, BN: int, NS: int) -> int:
    """PER, the blocks per split of the tile holding query row ``row`` of a causal self-attention sequence of ``T_seq`` rows in
    ``NS`` splits: ceil(NBM / NS), NBM the tile's key blocks (`block.h get_n_block_min_max`, seqlen_k = seqlen_q)."""
    if not 0 <= row < T_seq or NS < 1:
        raise ValueError(f"split_per: row {row} of {T_seq}, {NS} splits")
    nbm = min(-(-T_seq // BN), -(-((row // BM + 1) * BM) // BN))
    return -(-nbm // NS)


def row_splits(T: int, BN: int, PER: int) -> list[tuple[int, int]]:
    """The splits a row of T keys sees, as [b0, b1) block ranges in split order: [s PER, min((s + 1) PER, ceil(T / BN)))."""
    if T < 1 or PER < 1:
        raise ValueError(f"row_splits: T={T}, PER={PER}")
    nb = -(-T // BN)
    return [(b0, min(b0 + PER, nb)) for b0 in range(0, nb, PER)]


def check_combine_bound(static: int) -> None:
    """The launch's split bound against the combine instantiation `SparseMlaCombine` states (kLogMaxSplits 5)."""
    if static > W.LANES:
        raise ValueError(f"a launch bound of {static} splits runs FlashAttnFwdCombine at kLogMaxSplits > 5 (more than "
                         f"{W.LANES} lanes), which SparseMlaCombine does not state")


def head_split(B, S, tile_of, end, combine_of, zero, q, k, v, *, per: int | None = None, finish=None,
               name: str = "AttentionHeadSplit"):
    """`AttentionHeadSplit`'s body: per split of `row_splits(T, BN, PER)` its blocks last to first (``tile_of(nvis, first,
    check)``), ``end`` and then ``combine_of(nsplit)`` over the splits; ``zero(width)`` a zero word.  ``per`` replaces S.PER,
    ``finish(lses, parts)``, given, replaces the combine, and ``name`` heads a refusal."""
    T, D, BN, MF = S.T, S.D, S.BN, S.MASKED_FROM
    PER = S.PER if per is None else per
    nb = -(-T // BN)
    if not 0 <= MF <= nb - 1:
        raise ValueError(f"{name}: MASKED_FROM {MF} outside blocks 0..{nb - 1}")
    splits = row_splits(T, BN, PER)
    if len(splits) > W.LANES:
        raise ValueError(f"{name}: {len(splits)} splits, the combine's warp holds {W.LANES}")
    zero_row = None
    lses, parts = [], []
    for b0, b1 in splits:
        m = l = o = None
        for b in range(b1 - 1, b0 - 1, -1):
            lo, hi = b * BN, min(T, (b + 1) * BN)
            kb, vb = k[lo:hi], v[lo:hi]
            if hi - lo < BN:  # Clear_OOB_MN: the out-of-bounds rows are zero; their columns are masked and never read
                if zero_row is None:
                    z16 = zero(16)
                    zero_row = Coll(Array(D, z16.type), concat([z16.refs] * D))
                pad = [zero_row.refs] * (BN - hi + lo)
                kb = Coll(Array(BN, kb.type.elem), concat([kb.refs] + pad))
                vb = Coll(Array(BN, vb.type.elem), concat([vb.refs] + pad))
            first = b == b1 - 1
            blk = tile_of(hi - lo, first, first or b >= MF)
            if first:
                z = zero(32)
                st = B.call(blk, q, kb, vb, z, Coll(Array(4, z.type), concat([z.refs] * 4)), Coll(Array(D, z.type), concat([z.refs] * D)))
            else:
                st = B.call(blk, q, kb, vb, m, l, o)
            m, l, o = st[0], st[1], st[2]
        e = B.call(end, m, l, o)
        parts.append(e[0])
        lses.append(e[1])
    if finish is not None:
        return finish(lses, parts)
    return B.call(combine_of(len(parts)), array_of(lses), array_of(parts))


def attention(B, S, head, q, kc, vc, width: int):
    """`AttentionSplit`'s body: GQA over ``head``, head h reading KV head h // (NH / KVH); ``width`` the leaves of one bf16."""
    T, NH, KVH, D = S.T, S.NH, S.KVH, S.D
    G, Wd = NH // KVH, width * D
    outs = []
    for g in range(KVH):
        kg = Coll(Array(T, kc.type.elem.elem), strided_view(kc, g * Wd, Wd, KVH * Wd, 1, T * Wd))
        vg = Coll(Array(T, vc.type.elem.elem), strided_view(vc, g * Wd, Wd, KVH * Wd, 1, T * Wd))
        outs.append(B.batch(head, q[g * G:(g + 1) * G], kg, vg, axes=(0, None, None)))
    return Coll(Array(NH, q.type.elem), concat([o.refs for o in outs]))


def _head_type(S, bf16=BF16):
    return ((("q", Array(S.D, bf16)), ("k", Array(S.T, Array(S.D, bf16))), ("v", Array(S.T, Array(S.D, bf16)))), Array(S.D, bf16))


def _attention_type(S, bf16=BF16):
    return ((("q", Array(S.NH, Array(S.D, bf16))), ("kc", Array(S.T, Array(S.KVH, Array(S.D, bf16)))),
             ("vc", Array(S.T, Array(S.KVH, Array(S.D, bf16))))), Array(S.NH, Array(S.D, bf16)))


def _lse_head_type(S, bf16=BF16, f32=F32):
    return _head_type(S, bf16)[0], Tuple(Array(S.D, bf16), f32)


def _word_steps(B, S):
    """The word heads' `head_split` steps: (``tile_of``, the split end, ``zero``)."""
    def tile_of(nvis, first, check):
        return bind(AttnBlockV4, D=S.D, NVIS=nvis, FIRST=first, BN=S.BN, DOT=S.DOT, CHECK=check)
    end = bind(W.SparseMlaSplitEnd, DV=S.D, INV=S.INV, SCALE=S.D ** -0.5)
    zeros = {16: lambda: B.call(ZERO16), 32: lambda: B.call(ZERO32)}
    return tile_of, end, lambda w: zeros[w]()


@composite("AttentionHeadSplit", 1, ["T", "D", "BN", "DOT", "INV", "MASKED_FROM", "PER"], _head_type, conformance=_CONF)
def AttentionHeadSplit(B, S, q, k, v):
    """One query row at position T-1 of FA3's split causal prefill over keys 0..T-1: per split of `row_splits(T, BN, PER)` its
    `AttnBlock_v4`s last to first (the split's first visited one FIRST, block b with `CHECK = FIRST or b >= MASKED_FROM`),
    `SparseMlaSplitEnd_v1{DV=D, INV, SCALE=D^-0.5}`, then `SparseMlaCombine_v1{NSPLIT, DV=D, INV}` over the row's splits."""
    tile_of, end, zero = _word_steps(B, S)
    return head_split(B, S, tile_of, end, lambda n: bind(W.SparseMlaCombine, NSPLIT=n, DV=S.D, INV=S.INV), zero, q, k, v)


@composite("AttentionHeadLse", 1, ["T", "D", "BN", "DOT", "INV", "MASKED_FROM"], _lse_head_type, conformance=_LSE_CONF)
def AttentionHeadLse(B, S, q, k, v):
    """`AttentionHead_v4{T, D, BN, DOT, INV, MASKED_FROM}` with its lse, the pair FA3's unsplit kernel writes: the row's
    `AttnBlock_v4`s last to first (block b with `CHECK = b >= MASKED_FROM`), `SparseMlaSplitEnd_v1{DV=D, INV, SCALE=D^-0.5}`,
    then (bf16 of the partial, the lse)."""
    tile_of, end, zero = _word_steps(B, S)

    def finish(lses, parts):
        return tuple_of(B.batch(P.F2fpBf16, parts[0], axes=(0,)), lses[0])
    return head_split(B, S, tile_of, end, None, zero, q, k, v, per=-(-S.T // S.BN), finish=finish, name="AttentionHeadLse")


@composite("AttentionHeadSplitLse", 1, ["T", "D", "BN", "DOT", "INV", "MASKED_FROM", "PER"], _lse_head_type, conformance=_LSE_CONF)
def AttentionHeadSplitLse(B, S, q, k, v):
    """`AttentionHeadSplit_v1{T, D, BN, DOT, INV, MASKED_FROM, PER}` with its lse: the same splits, combined by
    `SparseMlaCombineLse_v1{NSPLIT, DV=D, INV}`."""
    tile_of, end, zero = _word_steps(B, S)
    return head_split(B, S, tile_of, end, lambda n: bind(W.SparseMlaCombineLse, NSPLIT=n, DV=S.D, INV=S.INV), zero, q, k, v,
                      name="AttentionHeadSplitLse")


@composite("AttentionSplit", 1, ["T", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM", "PER"], _attention_type, conformance=_CONF)
def AttentionSplit(B, S, q, kc, vc):
    """`Attention_v4`'s GQA over `AttentionHeadSplit_v1{T, D, BN, DOT, INV, MASKED_FROM, PER}`: every head of the row shares its
    tile, so one MASKED_FROM and one PER."""
    head = bind(AttentionHeadSplit, T=S.T, D=S.D, BN=S.BN, DOT=S.DOT, INV=S.INV, MASKED_FROM=S.MASKED_FROM, PER=S.PER)
    return attention(B, S, head, q, kc, vc, 1)


FA3_SPLIT_COMPOSITES = [AttentionHeadSplit, AttentionSplit, AttentionHeadLse, AttentionHeadSplitLse]

__all__ = ["AttentionHeadLse", "AttentionHeadSplit", "AttentionHeadSplitLse", "AttentionSplit", "FA3_SPLIT_COMPOSITES",
           "check_combine_bound", "row_splits", "split_per"]
