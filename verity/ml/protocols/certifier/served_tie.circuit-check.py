"""circuit-check's bindings of `verity.ml.protocols.certifier.served_tie`'s Definitions, in `circuit_check.targets'` forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_served_tie_roots() -> list[Any]:
    """The tie at the compile slice's class: corpus-smallkv's window of seven steps at R 2, its engine's L 4096 and T 256,
    request ids of 6 bits, and the port's i64 ids and positions (`tests/fixtures/compile_slice_smallkv.json`), as one
    unit (G = P) and as seven (G = 1, the block's selection), with the step and its slots, routing and scan."""
    from verity.ml.circuits.defs import bind
    from verity.ml.protocols.certifier.served_tie import (ServedTieScan, ServedTieSelect, ServedTieSlots, ServedTieStep,
                                                            unit)

    return [
        bind(ServedTieSlots, R=2, L=4096, T=256),
        bind(ServedTieScan, R=2, L=4096, T=256, IB=8, PB=8),
        bind(ServedTieStep, R=2, L=4096, T=256, IB=8, PB=8),
        bind(ServedTieSelect, P=7, G=1, SB=46),
        unit(R=2, P=7, G=7, IW=6, L=4096, T=256, IB=8, PB=8),
        unit(R=2, P=7, G=1, IW=6, L=4096, T=256, IB=8, PB=8),
    ]
