"""circuit-check's bindings of `verity.core.protocols.certifier.io_request`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_io_request_roots() -> list[Any]:
    """The ingress tie's prototype (`verity.core.protocols.certifier.io_request`) at the slow test's size class: a
    link-window of T = 4 buckets of R = 2 512-byte cells (N = 8, the view's prompt lengths routed over M = 8
    positions) with the own frame, which reaches both readings of a request frame, the scan's step inside a bucket
    and at its start, the own frame, the totals and the unit at their smallest; and the frame at B = 4,096, the served
    frame size."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.io_request import (IoTieRequest, IoTieRequestEnd, IoTieRequestFrame,
                                                            IoTieRequestOwn, IoTieRequestStep)

    return [
        bind(IoTieRequestFrame, B=512),
        bind(IoTieRequestFrame, B=4096),
        bind(IoTieRequestStep, D=3, NEW=0),
        bind(IoTieRequestStep, D=3, NEW=1),
        bind(IoTieRequestOwn, B=512),
        bind(IoTieRequestEnd),
        bind(IoTieRequest, B=512, T=4, R=2, OWN=1),
    ]
