"""circuit-check's bindings of `verity.definitions.registry.boolean_activation`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The final-logit softcap's tanh row at N = 8."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_activation as BA

    return [
        bind(BA.Bf16Tanh, N=8),
    ]
