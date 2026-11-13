"""circuit-check's bindings of `verity.definitions.registry.boolean_dsa_topk_mask`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

#: The Boolean Definitions of `dsa_topk_mask`'s sidecar's RANGES, over the same size classes and at the same spots, each held to
#: its word Definition there by its word view (`circuit_check.ranges`; ``circuit-check --as-call --ranges DsaTopKMask_v2 ...``).
#: The rows' read spans the row width too: at every W it is one network (`dsa_topk_mask.rows`, by N and K) of
#: `DsaWaksmanSwap_v2{W,E}` and `DsaWaksmanPick_v2{W,E}`, one AND a payload bit, so the spots check the network at every N at
#: one word and the latent rows' 576 words at the first class, where they are already over circuit-check's input budget and
#: read lazily (`circuit_check.targets.LAZY_OPERANDS`)
RANGES = {
    "DsaTopKMask_v2{N=4096..131072 step 2048,TOPK=2048}": (
        "N=4096", "N=131072",
        "N=6144", "N=10240", "N=18432", "N=34816", "N=67584", "N=8192", "N=129024",
    ),
    "DsaButterflySettings_v2{N=4096..131072 step 2048,K=2048}": (
        "N=4096", "N=131072",
        "N=6144", "N=10240", "N=18432", "N=34816", "N=67584", "N=8192", "N=129024",
    ),
    "DsaButterflyRows_v2{N=4096..131072 step 2048,K=2048,W=1..576,E=16}": (
        "N=4096,W=576", "N=4096,W=1", "N=131072,W=1",
        "N=6144,W=1", "N=10240,W=1", "N=18432,W=1", "N=34816,W=1", "N=67584,W=1", "N=8192,W=1", "N=129024,W=1",
    ),
    "DsaTopKRow_v2{N=4096..131072 step 2048,TOPK=2048}": (
        "N=4096", "N=131072",
        "N=6144", "N=10240", "N=18432", "N=34816", "N=67584", "N=8192", "N=129024",
    ),
    "DsaTopKMaskCounted_v2{N=4096..131072 step 2048,TOPK=2048}": (
        "N=4096", "N=131072",
        "N=6144", "N=10240", "N=18432", "N=34816", "N=67584", "N=8192", "N=129024", "N=40960", "N=43008",
    ),
    # `dsa_topk_mask`'s sidecar's RANGES on bits, over the same rows below their size class and at the same spots, each held to its
    # word Definition there by its word view (``circuit-check --as-call --ranges DsaButterflyRowsPad_v2``)
    "DsaButterflyRowsPad_v2{T=2049..131071,N=4096..131072 step 2048,K=2048,W=1..576,E=16}": (
        "T=2049,N=4096,W=576", "T=131071,N=131072,W=1",
        "T=4095,N=4096,W=1",
    ),
}


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The mask over 20 keys selecting 16 (reaching `DsaTopKKey_v2`, `DsaRadixPass_v1` and `DsaRadixTies_v1`), the counted mask
    over them (reaching `DsaRowValid_v1` at a row length no power of two) and the long row over them (the mask, its settings and
    the index read); the settings at small statics shaped as GLM-5.3's (an odd N,
    K = N - 1; reaching `DsaButterflyTagBits_v1` and `DsaButterflyCtl_v1`) and with picks (N 33, K 8); the rows' read likewise
    (N 9, K 8; N 7, K 3) and at GLM-5.3's row width (576 bf16) over 5 rows; and 6 rows padded to 9, and 9 padded to 16, a long
    row's shape (T >= K, N = 2K)."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_dsa_topk_mask as BM

    return [
        ("DsaTopKMask_v2", lambda: bind(BM.DsaTopKMaskV2, N=20, TOPK=16)),
        ("DsaTopKMaskCounted_v2", lambda: bind(BM.DsaTopKMaskCountedV2, N=20, TOPK=16)),
        ("DsaTopKRow_v2", lambda: bind(BM.DsaTopKRowV2, N=20, TOPK=16)),
        ("DsaButterflySettings_v2", lambda: bind(BM.DsaButterflySettingsV2, N=9, K=8)),
        ("DsaButterflySettings_v2-pick", lambda: bind(BM.DsaButterflySettingsV2, N=33, K=8)),
        ("DsaButterflyRows_v2", lambda: bind(BM.DsaButterflyRowsV2, N=9, K=8, W=3, E=16)),
        ("DsaButterflyRows_v2-pick", lambda: bind(BM.DsaButterflyRowsV2, N=7, K=3, W=2, E=16)),
        ("DsaButterflyRows_v2-glm", lambda: bind(BM.DsaButterflyRowsV2, N=5, K=4, W=576, E=16)),
        ("DsaButterflyRowsPad_v2", lambda: bind(BM.DsaButterflyRowsPadV2, T=6, N=9, K=8, W=3, E=16)),
        ("DsaButterflyRowsPad_v2-long", lambda: bind(BM.DsaButterflyRowsPadV2, T=9, N=16, K=8, W=3, E=16)),
    ]


def _ties_calls_roots() -> list[Any]:
    """The ties with their prefix count as Calls (`DsaRadixTies_v2`, reaching `DsaCountSum_v1`, `DsaCountSplit_v1` and
    `DsaRadixTie_v1` at each width): over the 20 keys of the mask above, a row length no power of two, and over 16; and at the
    long row's served statics, TOPK 2,048 over a block of 16,384 keys and over the longest row a unit masks whole, 40,960,
    each held to its digest pin."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_dsa_topk_mask as BM

    return [bind(BM.DsaRadixTiesV2, N=n, TOPK=k) for n, k in ((20, 16), (16, 8), (16384, 2048), (40960, 2048))]


def _tags_count_roots() -> list[Any]:
    """The tags and the count as one tree of counts (`DsaButterflyTagsCount_v1`), at rows of a power of two, off one and past
    one, with K below the row, at it and a power of two: reaching `DsaButterflyTag_v1` with and without a count."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_dsa_topk_mask as BM

    return [bind(BM.DsaButterflyTagsCount, N=n, K=k) for n, k in ((8, 4), (6, 4), (13, 13), (37, 9))]


def _select_calls_roots() -> list[Any]:
    """The radix select's pass and valid bits as Calls: `DsaRadixPass_v2` over the 20 keys of the mask above (under a chunk of
    its count), over two chunks and a rest (reaching `DsaRadixPassCount_v1` at both, `DsaCountSum_v1` over the chunks,
    `DsaRadixPassLeft_v1` and `DsaRadixPassKey_v1`) and at the longest row a unit masks whole, 40,960 keys at TOPK 2,048, held
    to its digest pin; `DsaRowValid_v2` at row lengths no power of two (reaching `DsaRowValidRoot_v1`, `DsaRowValidStep_v1`
    and `DsaRowValidLast_v1`) and at the served ones, the merge's 2,048 slots, a block's 16,384 keys and the whole row's
    40,960."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_dsa_topk_mask as BM

    return [*(bind(BM.DsaRadixPassV2, N=n, TOPK=k) for n, k in ((20, 16), (600, 512), (40960, 2048))),
            *(bind(BM.DsaRowValidV2, N=n) for n in (13, 20, 2048, 16384, 40960))]
