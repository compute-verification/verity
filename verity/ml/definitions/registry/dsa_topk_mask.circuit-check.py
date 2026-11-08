"""circuit-check's bindings of `verity.ml.definitions.registry.dsa_topk_mask`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

#: GLM-5.3's rows past INDEX_TOPK in the mask form at its TOPK = K = 2,048, over every key count from the first row past
#: INDEX_TOPK to the longest row at 128k context, with its latent rows (W = KV_LORA + QK_ROPE = 576 bf16 words) for the rows'
#: read, checked at these spots (`circuit_check.ranges`; ``circuit-check --as-call --ranges DsaTopKMask_v1 ...``).  Past 7,281
#: keys the rows are over circuit-check's input budget and read lazily (`circuit_check.targets.LAZY_OPERANDS`).  No kernel
#: realizes these word Definitions: their Boolean Definitions' word views compare them, at the same spots
#: (`boolean_dsa_topk_mask`'s sidecar)
RANGES = {
    "DsaTopKMask_v1{N=2049..131072,TOPK=2048}": (
        # the ends: the first row past INDEX_TOPK, and the longest row at 128k context
        "N=2049", "N=131072",
        # an even row, a power of two and one past it (the network's first row of 13 stages), the first rows of 15 and 17
        # stages, and a long row near no power of two
        "N=2050", "N=4096", "N=4097", "N=16385", "N=65537", "N=100001",
    ),
    "DsaButterflySettings_v1{N=2049..131072,K=2048}": (
        "N=2049", "N=131072",
        "N=2050", "N=4096", "N=4097", "N=16385", "N=65537", "N=100001",
    ),
    "DsaButterflyRows_v1{N=2049..131072,K=2048,W=576,E=16}": (
        "N=2049", "N=131072",
        "N=2050", "N=4096", "N=4097", "N=16385", "N=65537", "N=100001",
        # the last row whose rows are read whole and the first read lazily
        "N=7281", "N=7282",
    ),
    "DsaTopKRow_v1{N=2049..131072,TOPK=2048}": (
        "N=2049", "N=131072",
        "N=2050", "N=4096", "N=4097", "N=16385", "N=65537", "N=100001",
    ),
}


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The mask and the long row over 20 keys selecting 16 (reaching the threshold, slot and key accelerators), and the
    concentrator at small statics shaped as GLM-5.3's (an odd N, K = N - 1; reaching `DsaButterflyRoute_v1` and the tags' scan),
    with picks (N 33, K 8) and in the rows' read (N 7, K 3)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import dsa_topk_mask

    return [
        ("DsaTopKMask_v1", lambda: bind(dsa_topk_mask.DsaTopKMask, N=20, TOPK=16)),
        ("DsaTopKRow_v1", lambda: bind(dsa_topk_mask.DsaTopKRow, N=20, TOPK=16)),
        ("DsaButterflySettings_v1", lambda: bind(dsa_topk_mask.DsaButterflySettings, N=9, K=8)),
        ("DsaButterflySettings_v1-pick", lambda: bind(dsa_topk_mask.DsaButterflySettings, N=33, K=8)),
        ("DsaButterflyRows_v1", lambda: bind(dsa_topk_mask.DsaButterflyRows, N=9, K=8, W=3, E=16)),
        ("DsaButterflyRows_v1-pick", lambda: bind(dsa_topk_mask.DsaButterflyRows, N=7, K=3, W=2, E=16)),
    ]
