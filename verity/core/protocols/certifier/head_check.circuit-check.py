"""circuit-check's bindings of `verity.core.protocols.certifier.head_check`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_roots() -> list[Any]:
    """N2's head check (`verity.core.protocols.certifier.head_check`) for a 9-byte link name ("responses") at T = 4, with
    and without an occupancy profile, which reaches the opening, the fields and the counts at their smallest; and the
    count check at the deployed T = 600, 32·T − 1 = 19,199 ANDs."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.head_check import NetworkHeadCounts, unit

    return [
        unit(9, 4, True),
        unit(9, 4, False),
        bind(NetworkHeadCounts, T=600),
    ]
