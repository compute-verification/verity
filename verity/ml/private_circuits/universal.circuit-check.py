"""circuit-check's bindings of `verity.ml.private_circuits.universal`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _universal_roots() -> list[Any]:
    """Protocol 2's universal unit (`verity.ml.private_circuits.universal`) at two size classes: G = 16 gates over 16 inputs (W = 32 wires)
    with 8 outputs, and G = 64 over 64 (W = 128) with 32."""
    from verity.ml.circuits.defs import bind
    from verity.ml.private_circuits.universal import UniversalUnit

    return [
        bind(UniversalUnit, G=16, N_IN=16, N_OUT=8),
        bind(UniversalUnit, G=64, N_IN=64, N_OUT=32),
    ]
