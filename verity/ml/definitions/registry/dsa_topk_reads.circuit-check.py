"""circuit-check's bindings of `verity.ml.definitions.registry.dsa_topk_reads`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

#: the index units at the served statics (TOPK 2,048, blocks of `dsa_topk_reads.BLOCK` = 16,384 keys, a row's class N a
#: multiple of TOPK from 4,096 to 131,072): a block of N keys from AT, a last one of 2,048 to 16,384, and the merge of M blocks
RANGES = {
    "DsaMaskPositions_v1{N=2048..16384 step 2048,AT=0..114688 step 16384,K=2048}": (
        # a whole block first and last, the shortest last block after the first and last, the first class's one block and a
        # last block mid-row
        "N=16384,AT=0", "N=16384,AT=114688", "N=2048,AT=16384", "N=2048,AT=114688", "N=4096,AT=0", "N=10240,AT=49152",
    ),
    "DsaPositionsMerge_v1{M=1..8,K=2048}": ("M=1", "M=8", "M=3"),
}


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The index units at small statics (a first block and one from AT, the merge of one list and of three), and the long row
    over three blocks of 4 keys, two reads of a 16-row table."""
    from verity.ml.circuits.defs import bind
    from verity.ml.commitments.gates.embedding import read
    from verity.ml.definitions.registry import dsa_topk_reads as R

    return [
        ("DsaMaskPositions_v1", lambda: bind(R.DsaMaskPositions, N=8, AT=0, K=4)),
        ("DsaMaskPositions_v1-at", lambda: bind(R.DsaMaskPositions, N=6, AT=8, K=4)),
        ("DsaPositionsMerge_v1", lambda: bind(R.DsaPositionsMerge, M=1, K=4)),
        ("DsaPositionsMerge_v1-blocks", lambda: bind(R.DsaPositionsMerge, M=3, K=4)),
        ("DsaTopKReadRows_v1", lambda: bind(R.DsaTopKReadRows, N=12, TOPK=2, BLOCK=4, READ=read(16, 1))),
    ]


def _by_calls_roots() -> list[Any]:
    """`DsaTopKReadRows_v1` at the served statics, checked by its calls: its 2,048 reads are past what circuit-check builds
    whole.  The first class, the last whole one (`dsa_topk_mask_cut.WHOLE_MAX`), the first cut one and the longest row."""
    from verity.ml.circuits.defs import bind
    from verity.ml.commitments.gates.embedding import read
    from verity.ml.definitions.registry import dsa_topk_mask_cut as X
    from verity.ml.definitions.registry import dsa_topk_reads as R

    return [bind(R.DsaTopKReadRows, N=n, TOPK=2048, BLOCK=R.BLOCK, READ=read(1 << 17, 576))
            for n in (4096, X.WHOLE_MAX, X.WHOLE_MAX + 2048, 131072)]
