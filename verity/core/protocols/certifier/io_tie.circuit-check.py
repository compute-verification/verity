"""circuit-check's bindings of `verity.core.protocols.certifier.io_tie`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_io_tie_roots() -> list[Any]:
    """The I/O tie's prototype (`verity.core.protocols.certifier.io_tie`) at the tests' size class: one draw of a
    256-byte frame in a depth-7 frames tree with F = 32 outputs a request and C = 2 port rows (DT = 3), which reaches
    the frame read, the id event, the port row's hiding read and its column at their smallest; and the frame read and
    the id event at the served B = 4,096."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.io_tie import IoTieColumn, IoTieDraw, IoTieEvent, NetworkFrameRead, NetworkFrameSwap

    return [
        bind(NetworkFrameSwap),
        bind(IoTieColumn),
        bind(IoTieEvent, B=256),
        bind(IoTieEvent, B=4096),
        bind(NetworkFrameRead, B=256, D=2),
        bind(IoTieDraw, B=256, D=7, FOUT=32, C=2),
    ]
