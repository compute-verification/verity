"""circuit-check's bindings of `verity.protocols.accounting.memory.pous.prover.setup_circuit`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pous_roots() -> list[Any]:
    """PoUS's setup unit (`verity.protocols.accounting.memory.pous.prover.setup_circuit`) at the decode's W = 64 (C0 = 189, its
    144-byte base built inside from the salt and the block's place) over a 16-byte row: three slots of 6 payload bytes,
    so the slot's selection takes two merges with one leaf unpaired, the slot's bound is not a power of two, and the last
    slot's tail is zero; its base alone; and its ok word alone.  The unit's ANDs are past the whole lowering's budget, so the family's struct
    budget in the pins (``struct_max_gates``) has its structural lowering compared with the reference on the vectors."""
    from verity.circuits.defs import bind
    from verity.protocols.accounting.memory.pous.prover.setup_circuit import PousBitWord, PousP2SetupBase
    from verity.protocols.accounting.memory.pous.prover.setup_proof import prefix_head, setup_unit

    return [
        setup_unit(64, 189, 16),
        bind(PousP2SetupBase, W=64, C0=189, HEAD=prefix_head(64, 189).hex()),
        bind(PousBitWord),
    ]
