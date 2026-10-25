"""circuit-check's bindings of `verity.core.protocols.certifier.io_tie`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_io_tie_roots() -> list[Any]:
    """The I/O tie's prototype (`verity.core.protocols.certifier.io_tie`) at the tests' size class: a bucket row of
    R = 8 256-byte slots (four lanes of r = 2) and a port of S = 4 opened rows of P = 4 ids from a session of 2^3 steps
    (E = 16 entries routed over M = 16 positions), which reaches the cell's parse, the Beneš routing, the match against
    the map, the port's row reads and the bucket's unit at their smallest; the cell's parse at B = 4,096; and the
    ingress's rows root (`NetworkRowsRoot`) with the frame-v3 pad past its three rows."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.io_tie import (FrameV3Pad, IoTieBucket, IoTieCell, IoTieContent, IoTieMatch,
                                                        IoTiePort, IoTieRoute, NetworkRowsRoot)

    return [
        bind(IoTieCell, B=256),
        bind(IoTieCell, B=4096),
        bind(IoTieRoute, W=36, D=4),
        bind(IoTieMatch, N=8, D=4, E=16),
        bind(IoTieContent, B=256, R=8, P=4, S=4),
        bind(IoTiePort, P=4, S=4, DS=3),
        bind(FrameV3Pad, B=1),
        bind(NetworkRowsRoot, N_BITS=1024, COUNT=3),
        bind(IoTieBucket, B=256, R=8, P=4, S=4, DS=3),
    ]
