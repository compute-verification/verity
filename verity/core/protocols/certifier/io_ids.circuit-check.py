"""circuit-check's bindings of `verity.core.protocols.certifier.io_ids`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_io_ids_roots() -> list[Any]:
    """The ingress-ids tie's prototype (`verity.core.protocols.certifier.io_ids`): the unit at its smallest, a bucket
    row of L = 1 lane of R = 1 512-byte cell (NK = 45 slots, CB = 6-bit classes routed over 2^6 positions) and a Commit
    port of S = 2 rows of P = 2 input ids from a tree of depth 1 (the members sorted over 2^6 positions), which reaches
    both readings of a slot, the keys at a field and elsewhere and the chain's checks; the slots at B = 4,096, the
    served cell size; and the keys and the chain at the parts' tests' sizes. The slow test holds the unit at
    {512, 2, 2, 16, 8, 4} to its reference."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.io_ids import IoTieIdsChain, IoTieIdsKeys, IoTieIdsSlots, IoTieRequestIds

    return [
        bind(IoTieIdsSlots, B=512),
        bind(IoTieIdsSlots, B=4096),
        bind(IoTieIdsKeys, NK=5, CB=3, D1=3),
        bind(IoTieIdsChain, CB=3, D2=3),
        bind(IoTieRequestIds, B=512, L=1, R=1, P=2, S=2, DS=1),
    ]
