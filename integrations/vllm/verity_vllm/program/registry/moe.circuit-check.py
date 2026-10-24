"""circuit-check's bindings of `verity_vllm.program.registry.moe`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """GLM-5.3's sigmoid `noaux_tc` router (`MoeRouterSigmoidBias_v1`, the word view of `boolean_moe`'s) at 8 experts, 2
    slots, on both of its kernels."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import moe as M

    return [
        *[bind(d, E=8, TOPK=2, SCALE=2.5, KERNEL=k) for d in (M.MoeRouterSigmoidBias,)
          for k in M.ROUTER_KERNELS],
    ]
