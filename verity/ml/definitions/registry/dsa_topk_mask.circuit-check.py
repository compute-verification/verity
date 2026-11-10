"""circuit-check's bindings of `verity.ml.definitions.registry.dsa_topk_mask`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

#: GLM-5.3's rows past INDEX_TOPK in the mask form at its TOPK = K = 2,048: vLLM reads a row of more than INDEX_TOPK keys at its
#: size class, the next multiple of INDEX_TOPK, so the served key counts are the 63 classes from 4,096 to the longest row at 128k
#: context, with the latent rows (W = KV_LORA + QK_ROPE = 576 bf16 words) for the rows' read, checked at these spots
#: (`circuit_check.ranges`; ``circuit-check --as-call --ranges DsaTopKMask_v1 ...``).  No kernel realizes these word
#: Definitions: their Boolean Definitions' word views compare them, at the same spots (`boolean_dsa_topk_mask`'s sidecar).  The
#: rows' read spans the row width too: at every W it is one network (`dsa_topk_mask.rows`, by N and K) of
#: `DsaWaksmanSwap_v1{W,E}` and `DsaWaksmanPick_v1{W,E}`, so the spots check it at the latent rows' 576 words to 67,584 keys,
#: past 6,144 of them read lazily (`circuit_check.targets.LAZY_OPERANDS`), and at one word to 131,072, since the network
#: doesn't depend on W
RANGES = {
    "DsaTopKMask_v1{N=4096..131072 step 2048,TOPK=2048}": (
        # the ends: the first class and the longest row at 128k context
        "N=4096", "N=131072",
        # the network's first classes of 13 to 17 stages, a power of two (the first class whose latent rows are read lazily),
        # and the class below the top
        "N=6144", "N=10240", "N=18432", "N=34816", "N=67584", "N=8192", "N=129024",
    ),
    "DsaButterflySettings_v1{N=4096..131072 step 2048,K=2048}": (
        "N=4096", "N=131072",
        "N=6144", "N=10240", "N=18432", "N=34816", "N=67584", "N=8192", "N=129024",
    ),
    "DsaButterflyRows_v1{N=4096..131072 step 2048,K=2048,W=1..576,E=16}": (
        "N=4096,W=576", "N=131072,W=1",
        "N=6144,W=576", "N=10240,W=576", "N=18432,W=576", "N=34816,W=576", "N=67584,W=576", "N=8192,W=576", "N=129024,W=1",
    ),
    "DsaTopKRow_v1{N=4096..131072 step 2048,TOPK=2048}": (
        "N=4096", "N=131072",
        "N=6144", "N=10240", "N=18432", "N=34816", "N=67584", "N=8192", "N=129024",
    ),
    # B2a's long rows in class G's batched form, each selecting among its own n keys at its class N (`vllm_dsa.long_row_keys`),
    # whole up to `dsa_topk_mask_cut.WHOLE_MAX` = 40,960 keys and past it as its cut (`CUTS`), each unit held to the block: the
    # spots above, the last class checked whole and the first one cut (three blocks, the last of 10,240 keys)
    "DsaTopKMaskCounted_v1{N=4096..131072 step 2048,TOPK=2048}": (
        "N=4096", "N=131072",
        "N=6144", "N=10240", "N=18432", "N=34816", "N=67584", "N=8192", "N=129024", "N=40960", "N=43008",
    ),
    # GLM-5.3's rows past INDEX_TOPK = K = 2,048 read below their size class (`DsaButterflyRowsPad`): a row of T keys at its class N,
    # the next multiple of K, so N is one of the 63 classes 4,096..131,072 and T one of the N - 2,047..N - 1 keys below it (at T = N
    # the row is `DsaButterflyRows`).  A rule spans each static alone, so its T range is every class's together.  The spots are rows
    # vLLM serves: the first class's shortest (T = K + 1, the most pad) at the latent rows' 576 words, its rows and pad read lazily
    # (`circuit_check.targets.LAZY_OPERANDS`), and at one word its longest (one pad row) and the top class's longest.  At every W it
    # is `DsaButterflyRows{N}`'s one network over the rows and then the pad (``circuit-check --as-call --ranges
    # DsaButterflyRowsPad_v1``)
    "DsaButterflyRowsPad_v1{T=2049..131071,N=4096..131072 step 2048,K=2048,W=1..576,E=16}": (
        "T=2049,N=4096,W=576", "T=131071,N=131072,W=1",
        "T=4095,N=4096,W=1",
    ),
}


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The mask, the counted mask and the long row over 20 keys selecting 16 (reaching the thresholds, slot and key
    accelerators), and the concentrator at small statics shaped as GLM-5.3's (an odd N, K = N - 1; reaching `DsaButterflyRoute_v1`
    and the tags' scan), with picks (N 33, K 8) and in the rows' read (N 7, K 3); the rows' read of 6 rows padded to 9, and of 9
    padded to 16, a long row's shape (T >= K, N = 2K)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import dsa_topk_mask

    return [
        ("DsaTopKMask_v1", lambda: bind(dsa_topk_mask.DsaTopKMask, N=20, TOPK=16)),
        ("DsaTopKMaskCounted_v1", lambda: bind(dsa_topk_mask.DsaTopKMaskCounted, N=20, TOPK=16)),
        ("DsaTopKRow_v1", lambda: bind(dsa_topk_mask.DsaTopKRow, N=20, TOPK=16)),
        ("DsaButterflySettings_v1", lambda: bind(dsa_topk_mask.DsaButterflySettings, N=9, K=8)),
        ("DsaButterflySettings_v1-pick", lambda: bind(dsa_topk_mask.DsaButterflySettings, N=33, K=8)),
        ("DsaButterflyRows_v1", lambda: bind(dsa_topk_mask.DsaButterflyRows, N=9, K=8, W=3, E=16)),
        ("DsaButterflyRows_v1-pick", lambda: bind(dsa_topk_mask.DsaButterflyRows, N=7, K=3, W=2, E=16)),
        ("DsaButterflyRowsPad_v1", lambda: bind(dsa_topk_mask.DsaButterflyRowsPad, T=6, N=9, K=8, W=3, E=16)),
        ("DsaButterflyRowsPad_v1-long", lambda: bind(dsa_topk_mask.DsaButterflyRowsPad, T=9, N=16, K=8, W=3, E=16)),
    ]
