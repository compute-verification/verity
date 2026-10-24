"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_activation`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The final-logit softcap's tanh row at N = 8."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_activation as BA

    return [
        bind(BA.Bf16Tanh, N=8),
    ]
