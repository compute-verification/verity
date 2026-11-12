"""circuit-check's bindings of `verity.definitions.registry.dsa_topk_reads`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

#: the index units at the served statics (TOPK 2,048, blocks of `dsa_topk_reads.index_block` keys, a row's class N a
#: multiple of TOPK from 4,096 to 131,072): a block of N keys from AT (the whole row up to `dsa_topk_mask_cut.WHOLE_MAX`, then
#: 16,384 keys and a last one of 2,048 to 16,384), and the merge of M blocks
RANGES = {
    "DsaMaskPositions_v1{N=2048..40960 step 2048,AT=0..114688 step 16384,K=2048}": (
        # a whole block first and last, the shortest last block after the first and last, the first class's one block, a
        # last block mid-row, and the first and last whole rows past one block
        "N=16384,AT=0", "N=16384,AT=114688", "N=2048,AT=16384", "N=2048,AT=114688", "N=4096,AT=0", "N=10240,AT=49152",
        "N=18432,AT=0", "N=40960,AT=0",
    ),
    "DsaPositionsMerge_v1{M=1..8,K=2048}": ("M=1", "M=8", "M=3"),
}


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The index units at small statics (a first block and one from AT, the merge of one list and of three, chunks of two of
    four slots and of five, a slot of four and of five), the long row of 12 keys (its mask whole, so one block), two reads of
    a 16-row table, and the merge, chunk and slot the longest row calls (8 blocks, 2,048 slots in chunks of 32, N 2^17): the
    merge and the chunk over `targets.MAX_GATES`, each held to its digest pin."""
    from verity.circuits.defs import bind
    from verity.commitments.gates.embedding import read
    from verity.definitions.registry import dsa_topk_reads as R

    return [
        ("DsaMaskPositions_v1", lambda: bind(R.DsaMaskPositions, N=8, AT=0, K=4)),
        ("DsaMaskPositions_v1-at", lambda: bind(R.DsaMaskPositions, N=6, AT=8, K=4)),
        ("DsaPositionsMerge_v1", lambda: bind(R.DsaPositionsMerge, M=1, K=4)),
        ("DsaPositionsMerge_v1-blocks", lambda: bind(R.DsaPositionsMerge, M=3, K=4)),
        ("DsaPositionsMerge_v1-served", lambda: bind(R.DsaPositionsMerge, M=8, K=2048)),
        ("DsaPositionsChunk_v1", lambda: bind(R.DsaPositionsChunk, K=4, W=2)),
        ("DsaPositionsChunk_v1-odd", lambda: bind(R.DsaPositionsChunk, K=5, W=2)),
        ("DsaPositionsChunk_v1-served", lambda: bind(R.DsaPositionsChunk, K=2048, W=R.CHUNK)),
        ("DsaPositionsSlot_v1", lambda: bind(R.DsaPositionsSlot, K=4, N=20)),
        ("DsaPositionsSlot_v1-odd", lambda: bind(R.DsaPositionsSlot, K=5, N=12)),
        ("DsaPositionsSlot_v1-served", lambda: bind(R.DsaPositionsSlot, K=R.CHUNK, N=131072)),
        ("DsaTopKReadRows_v1", lambda: bind(R.DsaTopKReadRows, N=12, TOPK=2, BLOCK=4, READ=read(16, 1))),
    ]


def _merge_calls_roots() -> list[Any]:
    """The merge written as Calls at the traced one's bindings (one list, three, and the served 8 blocks of 2,048 slots, held
    to its digest pin), and its pieces, the two offsets at a small K and the served one."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import dsa_topk_reads as R

    return [*(bind(R.DsaPositionsMergeV2, M=m, K=k) for m, k in ((1, 4), (3, 4), (8, 2048))),
            bind(R.DsaPositionsKeep), bind(R.DsaPositionsShift), bind(R.DsaPositionsShiftIn),
            *(bind(d, K=k) for d in (R.DsaPositionsFirstOffset, R.DsaPositionsOffset) for k in (4, 2048))]


def _by_calls_roots() -> list[Any]:
    """`DsaTopKReadRows_v1` at the served statics, checked by its calls: its 2,048 reads are past what circuit-check builds
    whole.  The first class, the last whole one (`dsa_topk_mask_cut.WHOLE_MAX`), the first cut one and the longest row."""
    from verity.circuits.defs import bind
    from verity.commitments.gates.embedding import read
    from verity.definitions.registry import dsa_topk_mask_cut as X
    from verity.definitions.registry import dsa_topk_reads as R

    return [bind(R.DsaTopKReadRows, N=n, TOPK=2048, BLOCK=R.BLOCK, READ=read(1 << 17, 576))
            for n in (4096, X.WHOLE_MAX, X.WHOLE_MAX + 2048, 131072)]


def _positions_calls_roots() -> list[Any]:
    """The positions as Calls (`DsaMaskPositions_v2`) at the `_v1` units' bindings, which reach the tags and the count as one
    tree (`DsaButterflyTagsCount_v1`) and the runs of positions (`DsaIndexBits_v1`, `DsaPositionWord_v1`); a block off a
    multiple of its length; and the two the long row serves, a block of the cut and the longest row masked whole, each held to
    its digest pin."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import dsa_topk_mask_cut as X
    from verity.definitions.registry import dsa_topk_reads as R

    return [*(bind(R.DsaMaskPositionsV2, N=n, AT=at, K=k) for n, at, k in ((8, 0, 4), (6, 8, 4), (14, 0, 4), (37, 5, 9))),
            bind(R.DsaMaskPositionsV2, N=X.BLOCK, AT=X.BLOCK, K=2048), bind(R.DsaMaskPositionsV2, N=X.WHOLE_MAX, AT=0, K=2048)]
