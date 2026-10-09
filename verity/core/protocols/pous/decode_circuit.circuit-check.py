"""circuit-check's bindings of `verity.core.protocols.pous.decode_circuit`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pous_roots() -> list[Any]:
    """P2's decode of one block (`verity.core.protocols.pous.decode_circuit`) at W = 64 (C0 = 189, its 144-byte
    expansion base: one midstate block and a 16-byte rest) reaches the counter-mode SHA-512 and field pieces."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pous.decode_circuit import PousP2Decode

    return [
        bind(PousP2Decode, W=64, C0=189, L=144),
    ]
