"""circuit-check's bindings of `verity.ml.definitions.registry.dsa_topk_mask`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


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
