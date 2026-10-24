"""circuit-check's bindings of `verity.core.protocols.certifier.io_tie`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_io_tie_roots() -> list[Any]:
    """The I/O tie's prototype (`verity.core.protocols.certifier.io_tie`) at the tests' size class: a link-window of
    T = 4 buckets of R = 3 256-byte cells (N = 12 cells routed over M = 16 slots), which reaches the cell's parse, the
    Beneš routing, the match against the table, the frame-v3 pads past N in the cells' tree and both rows roots at
    their smallest; and the cell's parse at B = 4,096."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.io_tie import (FrameV3Pad, IoTieCell, IoTieContent, IoTieMatch, IoTieRoute,
                                                        IoTieWindow, NetworkRowsRoot)

    return [
        bind(IoTieCell, B=256),
        bind(IoTieCell, B=4096),
        bind(IoTieRoute, W=71, D=4),
        bind(IoTieMatch, D=4, DB=2),
        bind(IoTieContent, B=256, T=4, R=3),
        bind(FrameV3Pad, B=1),
        bind(NetworkRowsRoot, N_BITS=1024, COUNT=3),
        bind(IoTieWindow, B=256, T=4, R=3),
    ]
