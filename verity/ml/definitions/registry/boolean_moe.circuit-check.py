"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_moe`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """OLMoE's routed sum at TOPK = 8 over 8 columns, the sum in 64-column chunks (`MoeSum_v3`, GLM-5.3's at 96 chunks) at two
    chunks and a remainder of 8, OLMoE's router at 8 experts, 2 slots, 4 values per thread, and GLM-5.3's sigmoid `noaux_tc`
    router at 8 experts, 2 slots, on both of its kernels, on bits (its word view is `registry.moe.MoeRouterSigmoidBias_v1`)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_moe as BMOE
    from verity.ml.definitions.registry import moe as M

    return [
        bind(BMOE.MoeSum, TOPK=8, H=8),
        bind(BMOE.MoeSumV3, TOPK=8, H=136),
        bind(BMOE.MoeRouterTopKOrdered, E=8, TOPK=2, VPT=4),
        *[bind(d, E=8, TOPK=2, SCALE=2.5, KERNEL=k) for d in (BMOE.MoeRouterSigmoidBias,)
          for k in M.ROUTER_KERNELS],
    ]
