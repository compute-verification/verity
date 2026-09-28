"""circuit-check's bindings of `verity.protocols.accounting.space.pous.decode_circuit`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pous_roots() -> list[Any]:
    """P2's decode of one block (`verity.protocols.accounting.space.pous.decode_circuit`) at W = 64 (C0 = 189, its 116-byte
    expansion base) reaches the Keccak and field pieces."""
    from verity.primitives.circuits.defs import bind
    from verity.protocols.accounting.space.pous.decode_circuit import PousP2Decode

    return [
        bind(PousP2Decode, W=64, C0=189, L=116),
    ]
