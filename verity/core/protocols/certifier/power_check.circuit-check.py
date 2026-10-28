"""circuit-check's bindings of `verity.core.protocols.certifier.power_check`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _power_roots() -> list[Any]:
    """The power certifier's check for an 8-byte channel name ("outlet-3"), three counted kinds (declared MACs,
    compressions, forming units) and depth-2 trees, the size class its tests prove, which reaches both openings, the
    fields, the threshold and the signed digest at their smallest; and the threshold and the fields alone, so their
    lowerings are pinned apart from the commitment gadgets'."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.power_check import PowerHeadFields, PowerThreshold, unit

    return [
        unit(8, 3, 2, 2),
        bind(PowerThreshold, J=3),
        bind(PowerHeadFields, L=8),
    ]
