"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_dsa_topk_mask`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

#: The Boolean Definitions of `dsa_topk_mask`'s sidecar's RANGES, over the same key counts and at the same spots, each held to
#: its word Definition there by its word view (`circuit_check.ranges`; ``circuit-check --as-call --ranges DsaTopKMask_v2 ...``).
#: The rows' read spans the row width too: at every W it is one network (`dsa_topk_mask.rows`, by N and K) of
#: `DsaWaksmanSwap_v2{W,E}` and `DsaWaksmanPick_v2{W,E}`, one AND a payload bit, so the spots check the network at every N at
#: one word and the latent rows' 576 words at the first row past INDEX_TOPK, where they are already over circuit-check's input
#: budget and read lazily (`circuit_check.targets.LAZY_OPERANDS`)
RANGES = {
    "DsaTopKMask_v2{N=2049..131072,TOPK=2048}": (
        "N=2049", "N=131072",
        "N=2050", "N=4096", "N=4097", "N=16385", "N=65537", "N=100001",
    ),
    "DsaButterflySettings_v2{N=2049..131072,K=2048}": (
        "N=2049", "N=131072",
        "N=2050", "N=4096", "N=4097", "N=16385", "N=65537", "N=100001",
    ),
    "DsaButterflyRows_v2{N=2049..131072,K=2048,W=1..576,E=16}": (
        "N=2049,W=576", "N=2049,W=1", "N=131072,W=1",
        "N=2050,W=1", "N=4096,W=1", "N=4097,W=1", "N=16385,W=1", "N=65537,W=1", "N=100001,W=1",
    ),
    "DsaTopKRow_v2{N=2049..131072,TOPK=2048}": (
        "N=2049", "N=131072",
        "N=2050", "N=4096", "N=4097", "N=16385", "N=65537", "N=100001",
    ),
}


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The mask over 20 keys selecting 16 (reaching `DsaTopKKey_v2`, `DsaRadixPass_v1` and `DsaRadixTies_v1`) and the long row
    over them (the mask, its settings and the index read); the settings at small statics shaped as GLM-5.3's (an odd N,
    K = N - 1; reaching `DsaButterflyTagBits_v1` and `DsaButterflyCtl_v1`) and with picks (N 33, K 8); the rows' read likewise
    (N 9, K 8; N 7, K 3) and at GLM-5.3's row width (576 bf16) over 5 rows."""
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
    ]
