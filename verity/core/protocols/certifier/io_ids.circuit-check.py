"""circuit-check's bindings of `verity.core.protocols.certifier.io_ids`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_io_ids_roots() -> list[Any]:
    """The ingress-ids tie (`verity.core.protocols.certifier.io_ids`): the unit at its smallest, IoTieRequest's
    link-window of T = 2 buckets of R = 1 512-byte cell with the own frame (one request cell: NK = 45 slots, CB = 6-bit
    classes routed over 2^6 positions) and S = 2 opened steps of a P = 2-token port of i64 ids, one chunk each (the
    members sorted over 2^6 positions), which reaches both readings of a slot, the keys at a field and elsewhere, the
    ports' high bits and the chain's checks; the slots at B = 4,096, the served frame size; the port at the parts'
    test's size and at the served token budget P = 2,048 (CB = 11); and the keys and the chain at the parts' tests'
    sizes. The slow test holds the unit at {512, 4, 2, 1, 16, 8, 8} to its reference."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.io_ids import (IoTieIdsChain, IoTieIdsKeys, IoTieIdsPort, IoTieIdsSlots,
                                                        IoTieRequestIds)

    return [
        bind(IoTieIdsSlots, B=512),
        bind(IoTieIdsSlots, B=4096),
        bind(IoTieIdsKeys, NK=5, CB=3, D1=3),
        bind(IoTieIdsChain, CB=3, D2=3),
        bind(IoTieIdsPort, P=3, IB=8, CB=3),
        bind(IoTieIdsPort, P=2048, IB=8, CB=11),
        bind(IoTieRequestIds, B=512, T=2, R=1, OWN=1, P=2, S=2, IB=8),
    ]
