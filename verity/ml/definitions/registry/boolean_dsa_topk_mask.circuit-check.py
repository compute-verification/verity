"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_dsa_topk_mask`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

#: `dsa_topk_mask`'s sidecar's RANGES on bits, over the same rows below their size class and at the same spots, each held to its
#: word Definition there by its word view (``circuit-check --as-call --ranges DsaButterflyRowsPad_v2``)
RANGES = {
    "DsaButterflyRowsPad_v2{T=2049..131071,N=4096..131072 step 2048,K=2048,W=1..576,E=16}": (
        "T=2049,N=4096,W=576", "T=131071,N=131072,W=1",
        "T=4095,N=4096,W=1",
    ),
}


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The mask over 20 keys selecting 16 (reaching `DsaTopKKey_v2`, `DsaRadixPass_v1` and `DsaRadixTies_v1`) and the long row
    over them (the mask, its settings and the index read); the settings at small statics shaped as GLM-5.3's (an odd N,
    K = N - 1; reaching `DsaButterflyTagBits_v1` and `DsaButterflyCtl_v1`) and with picks (N 33, K 8); the rows' read likewise
    (N 9, K 8; N 7, K 3) and at GLM-5.3's row width (576 bf16) over 5 rows; and 6 rows padded to 9, and 9 padded to 16, a long
    row's shape (T >= K, N = 2K)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_dsa_topk_mask as BM

    return [
        ("DsaTopKMask_v2", lambda: bind(BM.DsaTopKMaskV2, N=20, TOPK=16)),
        ("DsaTopKRow_v2", lambda: bind(BM.DsaTopKRowV2, N=20, TOPK=16)),
        ("DsaButterflySettings_v2", lambda: bind(BM.DsaButterflySettingsV2, N=9, K=8)),
        ("DsaButterflySettings_v2-pick", lambda: bind(BM.DsaButterflySettingsV2, N=33, K=8)),
        ("DsaButterflyRows_v2", lambda: bind(BM.DsaButterflyRowsV2, N=9, K=8, W=3, E=16)),
        ("DsaButterflyRows_v2-pick", lambda: bind(BM.DsaButterflyRowsV2, N=7, K=3, W=2, E=16)),
        ("DsaButterflyRows_v2-glm", lambda: bind(BM.DsaButterflyRowsV2, N=5, K=4, W=576, E=16)),
        ("DsaButterflyRowsPad_v2", lambda: bind(BM.DsaButterflyRowsPadV2, T=6, N=9, K=8, W=3, E=16)),
        ("DsaButterflyRowsPad_v2-long", lambda: bind(BM.DsaButterflyRowsPadV2, T=9, N=16, K=8, W=3, E=16)),
    ]
