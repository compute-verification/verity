"""circuit-check's bindings of `verity_vllm.program.registry.boolean_moe`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """OLMoE's routed sum at TOPK = 8 over 8 columns and its router at 8 experts, 2 slots, 4 values per thread, and GLM-5.3's
    sigmoid `noaux_tc` router at 8 experts, 2 slots, on both of its kernels, on bits (its word view is
    `registry.moe.MoeRouterSigmoidBias_v1`)."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_moe as BMOE
    from verity_vllm.program.registry import moe as M

    return [
        bind(BMOE.MoeSum, TOPK=8, H=8),
        bind(BMOE.MoeRouterTopKOrdered, E=8, TOPK=2, VPT=4),
        *[bind(d, E=8, TOPK=2, SCALE=2.5, KERNEL=k) for d in (BMOE.MoeRouterSigmoidBias,)
          for k in M.ROUTER_KERNELS],
    ]
