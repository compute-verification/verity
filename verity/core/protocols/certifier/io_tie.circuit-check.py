"""circuit-check's bindings of `verity.core.protocols.certifier.io_tie`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_io_tie_roots() -> list[Any]:
    """The I/O tie's prototype (`verity.core.protocols.certifier.io_tie`) at the tests' size class: a bucket row of
    R = 8 256-byte slots (four lanes of r = 2) and a port of S = 4 read rows of P = 4 ids (E = 16 entries routed over
    M = 16 positions), which reaches the cell's parse, the Beneš routing (and its switch), the match against the map
    and the bucket's unit at their smallest; and the cell's parse at B = 4,096."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.io_tie import (IoTieBucket, IoTieCell, IoTieContent, IoTieMatch, IoTieRoute,
                                                        IoTieSwitch)

    return [
        bind(IoTieCell, B=256),
        bind(IoTieCell, B=4096),
        bind(IoTieSwitch, W=36),
        bind(IoTieRoute, W=36, D=4),
        bind(IoTieMatch, N=8, D=4, E=16),
        bind(IoTieContent, B=256, R=8, P=4, S=4),
        bind(IoTieBucket, B=256, R=8, P=4, S=4),
    ]
