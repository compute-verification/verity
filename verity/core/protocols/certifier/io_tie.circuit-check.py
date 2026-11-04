"""circuit-check's bindings of `verity.core.protocols.certifier.io_tie`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_io_tie_roots() -> list[Any]:
    """The Beneš routing the network certifier's ties share (`verity.core.protocols.certifier.io_tie`) at the tests'
    size class: a switch of W = 36 bits and the network over M = 16 positions of them."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.io_tie import IoTieRoute, IoTieSwitch

    return [
        bind(IoTieSwitch, W=36),
        bind(IoTieRoute, W=36, D=4),
    ]
