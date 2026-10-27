"""circuit-check's bindings of `verity.core.protocols.certifier.io_request`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_io_request_roots() -> list[Any]:
    """The ingress tie's prototype (`verity.core.protocols.certifier.io_request`) on a bucket row of the warden's
    ingress: the unit at the test's size class, L = 2 lanes of R = 2 512-byte cells, which reaches both readings of a
    request cell, a lane's scan from its carry-in and the carry-out at their smallest, and at one lane, whose verdict
    is its one lane's; the frame at B = 4,096, the served cell size, and the unit at the served link's class, one lane of
    one 4,096-byte cell; the unit at the gate's warden schedule (`warden-rehearse --scan`), 4 lanes of r = 4 512-byte
    cells; and the step and the end."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.io_request import (IoTieRequest, IoTieRequestEnd, IoTieRequestFrame,
                                                            IoTieRequestStep)

    return [
        bind(IoTieRequestFrame, B=512),
        bind(IoTieRequestFrame, B=4096),
        bind(IoTieRequestStep),
        bind(IoTieRequestEnd),
        bind(IoTieRequest, B=512, L=2, R=2),
        bind(IoTieRequest, B=512, L=1, R=2),
        bind(IoTieRequest, B=4096, L=1, R=1),
        bind(IoTieRequest, B=512, L=4, R=4),
    ]
