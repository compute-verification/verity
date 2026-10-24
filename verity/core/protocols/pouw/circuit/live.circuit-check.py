"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.live`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwLiveSlots_v1` at T = 8 and T = 16, two window slots (`h1window.slot`, a power of two of at least the tile's R):
    the m ≤ T test, T request liveness, index and prefix tests, the popcount N, and T slot tests against N and m. It grows
    linearly in T."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pouw.circuit import live as LV

    return [bind(LV.PouwLiveSlots, T=t) for t in (8, 16)]
