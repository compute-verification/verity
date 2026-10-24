"""circuit-check's bindings of `verity.core.protocols.certifier.io_ids`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_io_ids_roots() -> list[Any]:
    """The ingress-ids tie's prototype (`verity.core.protocols.certifier.io_ids`) at the slow test's size class:
    IoTieRequest's link-window of T = 4 buckets of R = 2 512-byte cells with the own frame (NK = 315 slots, CB = 9-bit
    classes routed over 2^9 positions) and a Commit port of S = 8 rows of P = 16 input ids from a tree of depth 4 (the
    members sorted over 2^9 positions), which reaches both readings of a slot, the keys at a field and elsewhere, the
    chain's checks and the unit at their smallest; the slots at B = 4,096, the served frame size; and the keys and the
    chain at the parts' tests' sizes."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.io_ids import IoTieIdsChain, IoTieIdsKeys, IoTieIdsSlots, IoTieRequestIds

    return [
        bind(IoTieIdsSlots, B=512),
        bind(IoTieIdsSlots, B=4096),
        bind(IoTieIdsKeys, NK=5, CB=3, D1=3),
        bind(IoTieIdsChain, CB=3, D2=3),
        bind(IoTieRequestIds, B=512, T=4, R=2, OWN=1, P=16, S=8, DS=4),
    ]
