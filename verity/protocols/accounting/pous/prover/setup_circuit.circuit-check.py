"""circuit-check's bindings of `verity.protocols.accounting.pous.prover.setup_circuit`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pous_roots() -> list[Any]:
    """PoUS's setup unit (`verity.protocols.accounting.pous.prover.setup_circuit`) at the decode's W = 64 (C0 = 189, its
    144-byte base) over a 15-byte row: three slots of 6 payload bytes, so the slot's selection takes two merges with one
    leaf unpaired, the slot's bound is not a power of two, and the last slot's tail is zero.  Its ANDs are past the whole
    lowering's budget, so the family's struct budget in the pins (``struct_max_gates``) has its structural lowering compared
    with the reference on the vectors."""
    from verity.circuits.defs import bind
    from verity.protocols.accounting.pous.prover.setup_circuit import PousP2SetupCheck

    return [
        bind(PousP2SetupCheck, W=64, C0=189, L=144, N=15),
    ]
