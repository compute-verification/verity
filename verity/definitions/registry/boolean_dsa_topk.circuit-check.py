"""circuit-check's bindings of `verity.definitions.registry.boolean_dsa_topk`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The canonical row's check over 20 keys selecting 16 (b = 5 index bits, 27 high bits required zero; reaches
    `DsaTopKOneHot_v1`, `DsaTopKVerdict_v1` and `DsaTopKAscending_v1`); the Waksman read at small statics shaped as GLM-5.3's
    (an odd N, K = N - 1), with a pick (N 6, K 3), and its index routing; and one switch at GLM-5.3's row width (576 bf16), the
    network's unit."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_dsa_topk

    return [
        ("DsaTopKLongRow_v2", lambda: bind(boolean_dsa_topk.DsaTopKLongRowV2, N=20, TOPK=16)),
        ("DsaWaksmanRows_v2", lambda: bind(boolean_dsa_topk.DsaWaksmanRowsV2, N=9, K=8, W=3, E=16)),
        ("DsaWaksmanRows_v2-pick", lambda: bind(boolean_dsa_topk.DsaWaksmanRowsV2, N=6, K=3, W=2, E=16)),
        ("DsaWaksmanOrder_v2", lambda: bind(boolean_dsa_topk.DsaWaksmanOrderV2, N=9, TOPK=8)),
        ("DsaWaksmanSwap_v2", lambda: bind(boolean_dsa_topk.DsaWaksmanSwapV2, W=576, E=16)),
    ]
