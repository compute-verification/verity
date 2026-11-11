"""circuit-check's bindings of `verity.definitions.registry.moe`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """GLM-5.3's served router (``served-``): `MoeRouterSigmoidBias_v1` at its 256 experts, top 8, scale 2.5, on each kernel
    `router_kernel` picks (block to 1,024 tokens, warp past them), which verity-vllm's `test_moe_fp8_served` holds to the
    compiler's choice."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import moe as M

    return [(f"MoeRouterSigmoidBias_v1/served-{k}", lambda k=k: bind(M.MoeRouterSigmoidBias, E=256, TOPK=8, SCALE=2.5, KERNEL=k))
            for k in M.ROUTER_KERNELS]


def _boolean_roots() -> list[Any]:
    """GLM-5.3's sigmoid `noaux_tc` router (`MoeRouterSigmoidBias_v1`, the word view of `boolean_moe`'s) and its one-field-a-slot
    form (`MoeRouterSigmoidBiasSlots_v1`), each on both of its kernels: at 8 experts, 2 slots, and at GLM-5.3's served statics
    (256 experts, 8 slots, the routed scaling factor 2.5), where serving picks the warp kernel at more than 1,024 tokens and the
    block kernel at fewer (`moe.router_kernel`)."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import moe as M

    return [
        *[bind(d, E=e, TOPK=topk, SCALE=2.5, KERNEL=k) for e, topk in ((8, 2), (256, 8))
          for d in (M.MoeRouterSigmoidBias, M.MoeRouterSigmoidBiasSlots) for k in M.ROUTER_KERNELS],
    ]
