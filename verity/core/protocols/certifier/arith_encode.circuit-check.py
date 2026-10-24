"""circuit-check's bindings of `verity.core.protocols.certifier.arith_encode`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_arith_encode_roots() -> list[Any]:
    """Advice v1's encoder (`verity.core.protocols.certifier.arith_encode`): the step at K = 18 (every symbol's
    scalings) for decisions (BIN = 1) and for general symbols (BIN = 0), and at K = 3, its smallest that reaches a
    refusal for too many scalings; the finish; and the model at plain contexts (C = 3, NW = 24) and at its smallest
    (C = 1, NW = 15)."""
    from verity.core.protocols.certifier.arith_encode import finish_unit, kt_unit, step_unit

    return [step_unit(18, 1), step_unit(18, 0), step_unit(3, 1), finish_unit(), kt_unit(3, 24), kt_unit(1, 15)]
