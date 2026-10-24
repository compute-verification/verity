"""circuit-check's bindings of `verity.core.protocols.certifier.advised`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_advised_roots() -> list[Any]:
    """N2's advised counts (`verity.core.protocols.certifier.advised`) at the tests' size classes (T = 8, R = 3,
    R_in = 2, unsalted and salted), which reach every part at its smallest, and at the deployed T = 600: R = 6 and
    R_in = 1, and R = 16 and R_in = 4 salted, as the served counts' (`served.circuit-check.py`)."""
    from verity.core.protocols.certifier.advised import unit

    return [
        unit(8, 3, 2, 0),
        unit(8, 3, 2, 1),
        unit(600, 6, 1, 0),
        unit(600, 16, 4, 1),
    ]
