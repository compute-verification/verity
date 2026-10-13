"""FA3's split-KV dense prefill (vLLM's served `num_splits = 0`): `AttnBlockFirst_v1`, `AttentionHeadSplit_v1`, `AttentionSplit_v1`.

Under `Split` (vllm-flash-attn 506341a1 `hopper/block.h::get_n_block_min_max`) a query tile's key blocks [0, NBM), NBM its
`n_block_max` (`fa3_model.kernel_geometry` with ``kblock_m``), are cut into its sequence's NS splits (`prepare_varlen_num_blocks`'
dynamic count, `fa3_model.split_plan`): split s runs the mainloop (`mainloop_fwd_sm90_tma_gmma_ws.hpp`, `mma()`) over [s * per,
min((s + 1) * per, NBM)), per = ceil(NBM / NS), its last block the first iteration (`Is_first`, `Check_inf`), then down, the
causal-masked iterations (at and above `MASKED_FROM`) with `Check_inf` and the rest without.  `Softmax::finalize` at `Split` writes
the split's partial output and lse (`SparseMlaSplitEnd_v1`) and `FlashAttnFwdCombine` merges a row's splits (`SparseMlaCombine_v1`).
An empty split (s * per >= NBM) writes lse -inf (`store_zero`), as does a split all of whose columns are past the row's causal
limit (the row max stays -inf, the sum 0), and the combine reads either exactly as one of its -inf pad lanes, so a row of T keys
reads `split_ranges`' splits (the tile's, cut at its ceil(T / BN) blocks) and the combine runs over those.  Blocks past the row's
limit that a split visits before its first visible one are neutral (`AttentionHead_v4`'s assumption: S = -inf, P = 0, the running
state zero, the next block's rescale of it 0), so a split's last visible block is its FIRST, always a `Check_inf` iteration.

The split end and the combine read no key, so the dense head takes GLM-SPLIT's (`sparse_mla_split`) as they are, at DV = D and
SCALE = D^-0.5.  The blocks are `AttnBlock_v4`'s; `AttnBlockFirst_v1` is its first iteration without the running state it does not
read (which `AttnBlock_v4{FIRST}` takes as inputs), so the head passes each split's first block no zero no gate reads.  A sequence
FA3 runs as one split is written unsplit (`AttentionHead_v4`): the split chain takes NS >= 2, and `split_of` gives the target's
`Attention_v4` binding back at NS = 1.  At NS >= 2 every row is combined, a row with one visible split too.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Coll, array_of, concat
from verity.ml.circuits.types import Array, Tuple
from verity_vllm.program.registry import fa3_check_inf as F3
from verity_vllm.program.registry import sparse_mla_split as W
from verity_vllm.program.registry.b1 import ZERO16
from verity_vllm.program.registry.prims import BF16, F32

_CONF = ("model-untested: read from vllm-flash-attn 506341a1's FA3 split path (hopper/block.h, mainloop_fwd_sm90_tma_gmma_ws.hpp, "
         "softmax.h) at hdim 256, its split end and combine sparse_mla_split's; tests/properties/glm53_fa3_prefill_capture_gpu.py "
         "is its hardware check")


def scale(D: int) -> float:
    """The softmax scale FA3's head-D prefill runs at, `AttnBlock_v4`'s (`b1._scale_log2_bits(D)` is its `scale_log2`)."""
    return int(D) ** -0.5


def split_ranges(T: int, NBM: int, NS: int, BN: int) -> list[tuple[int, int]]:
    """The key blocks [b0, e) a query row with T visible keys reads in each split it is not masked from, in split order: the tile's
    NS splits of its NBM blocks (`sparse_mla_split.split_blocks`), cut at the row's ceil(T / BN)."""
    nb = -(-int(T) // int(BN))
    return [(b0, min(b1, nb)) for b0, b1 in W.split_blocks(int(NBM) * int(BN), int(NS), int(BN)) if b0 < nb]


def check_split_statics(name: str, T: int, NBM: int, NS: int, BN: int, MASKED_FROM: int) -> None:
    nb = -(-int(T) // int(BN))
    if T < 1 or nb > NBM or not 2 <= NS <= W.LANES or not 0 <= MASKED_FROM <= nb - 1:
        raise ValueError(f"{name}: T={T} keys in {nb} blocks of {BN}, the tile's NBM={NBM} (at least {nb}), NS={NS} splits (2 to "
                         f"{W.LANES}; one split is AttentionHead_v4), MASKED_FROM={MASKED_FROM} (0 to {nb - 1})")


def head_split(B, S, first_of, block_of, end, combine_of, bf16, zero16, q, k, v):
    """The split head's body: each of `split_ranges`' splits, its blocks last to first (the first ``first_of(nvis)``, then
    ``block_of(b, nvis)``; the row's partial block zero-padded, as `AttentionHead_v4` pads it), ``end``, then ``combine_of(n)`` over
    the n splits."""
    T, D, BN = S.T, S.D, S.BN
    check_split_statics(f"AttentionHeadSplit{{T={T},NBM={S.NBM},NS={S.NS}}}", T, S.NBM, S.NS, BN, S.MASKED_FROM)
    zero_row = None
    lses, parts = [], []
    for b0, e in split_ranges(T, S.NBM, S.NS, BN):
        st = None
        for b in range(e - 1, b0 - 1, -1):
            lo, hi = b * BN, min(T, (b + 1) * BN)
            kb, vb = k[lo:hi], v[lo:hi]
            if hi - lo < BN:
                if zero_row is None:
                    zero_row = Coll(Array(D, bf16), concat([zero16().refs] * D))
                pad = [zero_row.refs] * (BN - hi + lo)
                kb = Coll(Array(BN, Array(D, bf16)), concat([kb.refs] + pad))
                vb = Coll(Array(BN, Array(D, bf16)), concat([vb.refs] + pad))
            st = (B.call(first_of(hi - lo), q, kb, vb) if st is None
                  else B.call(block_of(b, hi - lo), q, kb, vb, st[0], st[1], st[2]))
        r = B.call(end, st[0], st[1], st[2])
        parts.append(r[0])
        lses.append(r[1])
    return B.call(combine_of(len(parts)), array_of(lses), array_of(parts))


def _first_type(S, bf16=BF16, f32=F32):
    return ((("q", Array(S.D, bf16)), ("k", Array(S.BN, Array(S.D, bf16))), ("v", Array(S.BN, Array(S.D, bf16)))),
            Tuple(f32, Array(4, f32), Array(S.D, f32)))


@composite("AttnBlockFirst", 1, ["D", "NVIS", "BN", "DOT"], _first_type, conformance=_CONF)
def AttnBlockFirst(B, S, q, k, v):
    """`AttnBlock_v4{D, NVIS, FIRST = true, BN, DOT, CHECK = true}` without the running state it does not read: FA3's first
    iteration of a tile or a split (`Is_first`, `Check_inf`), S = Q K^T over NVIS visible keys, the row max, P = ex2(S * scale -
    guard(max) * scale), the per-lane sums and O = P V from zero."""
    return F3.AttnBlockV4.body(B, SimpleNamespace(D=S.D, NVIS=S.NVIS, FIRST=True, BN=S.BN, DOT=S.DOT, CHECK=True), q, k, v,
                               None, None, None)


def _head_type(S, bf16=BF16):
    return ((("q", Array(S.D, bf16)), ("k", Array(S.T, Array(S.D, bf16))), ("v", Array(S.T, Array(S.D, bf16)))), Array(S.D, bf16))


@composite("AttentionHeadSplit", 1, ["T", "NBM", "NS", "D", "BN", "DOT", "INV", "MASKED_FROM"], _head_type, conformance=_CONF)
def AttentionHeadSplit(B, S, q, k, v):
    """One query head at position T - 1 of a causal sequence FA3 runs in NS >= 2 splits, its tile's n_block_max NBM: each split
    `split_ranges` gives, its first visible block `AttnBlockFirst_v1` and the rest `AttnBlock_v4{..., CHECK = b >= MASKED_FROM}`,
    last to first, its `SparseMlaSplitEnd_v1{DV = D, INV, SCALE = D^-0.5}`, then `SparseMlaCombine_v1{NSPLIT, DV = D, INV}`."""
    return head_split(B, S, lambda nvis: bind(AttnBlockFirst, D=S.D, NVIS=nvis, BN=S.BN, DOT=S.DOT),
                      lambda b, nvis: bind(F3.AttnBlockV4, D=S.D, NVIS=nvis, FIRST=False, BN=S.BN, DOT=S.DOT,
                                           CHECK=b >= S.MASKED_FROM),
                      bind(W.SparseMlaSplitEnd, DV=S.D, INV=S.INV, SCALE=scale(S.D)),
                      lambda n: bind(W.SparseMlaCombine, NSPLIT=n, DV=S.D, INV=S.INV), BF16, lambda: B.call(ZERO16), q, k, v)


def _attention_type(S, bf16=BF16):
    return ((("q", Array(S.NH, Array(S.D, bf16))), ("kc", Array(S.T, Array(S.KVH, Array(S.D, bf16)))),
             ("vc", Array(S.T, Array(S.KVH, Array(S.D, bf16))))), Array(S.NH, Array(S.D, bf16)))


@composite("AttentionSplit", 1, ["T", "NBM", "NS", "NH", "KVH", "D", "BN", "DOT", "INV", "MASKED_FROM"], _attention_type,
           conformance=_CONF)
def AttentionSplit(B, S, q, kc, vc):
    """`Attention_v4` over `AttentionHeadSplit_v1`: one query row's NH heads (GQA, head h reads KV head h // (NH / KVH)), every
    head of the row in its tile, so one NBM and one MASKED_FROM, and in its sequence's NS splits."""
    return F3.AttentionV4.body(B, S, q, kc, vc, head=bind(AttentionHeadSplit, NBM=S.NBM, NS=S.NS))


def split_of(spec, NBM: int, NS: int):
    """The split chain of a row the target binds `Attention_v4` (`targets.attention_spec`) in a sequence FA3 runs in NS splits, its
    tile's n_block_max NBM: `AttentionSplit_v1` at the same statics, or the row's own binding at NS = 1 (written unsplit)."""
    if spec.base_id != "Attention_v4":
        raise ValueError(f"split_of: {spec.id} is not an Attention_v4 row (the split chain is FA3's per-iteration Check_inf one)")
    return spec if int(NS) == 1 else bind(AttentionSplit, **spec.bindings, NBM=int(NBM), NS=int(NS))


FA3_SPLIT_COMPOSITES = [AttnBlockFirst, AttentionHeadSplit, AttentionSplit]

__all__ = ["AttnBlockFirst", "AttentionHeadSplit", "AttentionSplit", "FA3_SPLIT_COMPOSITES", "check_split_statics", "head_split",
           "scale", "split_of", "split_ranges"]
