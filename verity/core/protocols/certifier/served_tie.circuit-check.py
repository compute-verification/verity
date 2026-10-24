"""circuit-check's bindings of `verity.core.protocols.certifier.served_tie`'s Definition, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_served_tie_roots() -> list[Any]:
    """The tie at the compile slice's class: corpus-smallkv's window of seven steps at R 2, its engine's L 4096 and T 256,
    request ids of 6 bits, and the port's i64 ids and positions (`tests/fixtures/compile_slice_smallkv.json`)."""
    from verity.core.protocols.certifier.served_tie import unit

    return [unit(R=2, P=7, IW=6, L=4096, T=256, IB=8, PB=8)]
