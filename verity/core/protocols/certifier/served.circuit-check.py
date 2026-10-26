"""circuit-check's bindings of `verity.core.protocols.certifier.served`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_served_roots() -> list[Any]:
    """N2's served counts (`verity.core.protocols.certifier.served`) at the tests' size classes (T = 8, F = 3, R = 2,
    unsalted and salted; and T = 12 with F_in = 2, salted), which reach every part at its smallest, and at the deployed
    T = 600 with F = 1024: one frame a request at R = 6, R_in = 1, and two at R = 16, R_in = 4, salted."""
    from verity.core.protocols.certifier.served import unit

    return [
        unit(8, 3, 1, 2, 2, 0),
        unit(8, 3, 1, 2, 2, 1),
        unit(12, 3, 2, 2, 2, 1),
        unit(600, 1024, 1, 6, 1, 0),
        unit(600, 1024, 2, 16, 4, 1),
    ]
