"""circuit-check's bindings of `verity_vllm.definitions.registry.boolean_tf32_wgmma`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    from verity.circuits.defs import bind

    from verity_vllm.definitions.registry import boolean_tf32_wgmma as BW

    return [("HopperTf32WgmmaDot8_v2", lambda: bind(BW.HopperTf32WgmmaDot8V2))]
