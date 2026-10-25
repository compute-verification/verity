"""circuit-check's bindings of `verity.ml.definitions.registry.dsa_topk`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The relation at GLM-5.3's first row past TOPK (2049 keys, TOPK 2048: one primitive over the row), and the Waksman read at
    small statics shaped as GLM-5.3's (an odd N, K = N - 1; reaching `DsaWaksmanSwap_v1`, `DsaWaksmanPick_v1` and both `Xor3x*`)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import dsa_topk

    return [
        ("DsaTopKOrder_v1", lambda: bind(dsa_topk.DsaTopKOrder, N=2049, TOPK=2048)),
        ("DsaWaksmanRows_v1", lambda: bind(dsa_topk.DsaWaksmanRows, N=9, K=8, W=3, E=16)),
        ("DsaWaksmanRows_v1-pick", lambda: bind(dsa_topk.DsaWaksmanRows, N=6, K=3, W=2, E=16)),
        ("DsaWaksmanOrder_v1", lambda: bind(dsa_topk.DsaWaksmanOrder, N=9, TOPK=8)),
    ]
