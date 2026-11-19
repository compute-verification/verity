"""circuit-check's bindings of `verity.protocols.resource_accounting.memory.pous.prover.setup_circuit`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pous_roots() -> list[Any]:
    """PoUS's setup unit (`verity.protocols.resource_accounting.memory.pous.prover.setup_circuit`) at the decode's W = 64 (C0 = 189, its
    144-byte base built inside from the salt and the block's place) over a 16-byte row: three slots of 6 payload bytes,
    so the slot's selection takes two merges with one leaf unpaired, the slot's bound is not a power of two, and the last
    slot's tail is zero; and its base alone.  The unit's ANDs are past the whole lowering's budget, so the family's struct
    budget in the pins (``struct_max_gates``) has its structural lowering compared with the reference on the vectors."""
    from verity.circuits.defs import bind
    from verity.protocols.resource_accounting.memory.pous.prover.setup_circuit import PousP2SetupBase, PousP2SetupCheck

    return [
        bind(PousP2SetupCheck, W=64, C0=189, N=16),
        bind(PousP2SetupBase, W=64, C0=189),
    ]
