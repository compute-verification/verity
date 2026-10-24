"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.live`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwLiveSlots_v1` at T = 32 and T = 96, two tables' leaf counts (the pouw setting ``rows``, a multiple of 32; 96's
    window slot is 128, so T < M there): the m ≤ T test, T request liveness, index and prefix tests, the popcount N, T
    slot tests against N and m, and each request's slot count against what it takes in round s. It grows as T², the
    counts."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pouw.circuit import live as LV

    return [bind(LV.PouwLiveSlots, T=t) for t in (32, 96)]
