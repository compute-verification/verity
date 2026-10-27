"""circuit-check's bindings of `verity.core.protocols.pous.verdict_circuit`'s Definition, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pous_roots() -> list[Any]:
    """PoUS's session verdict (`verity.core.protocols.pous.verdict_circuit`) at the audit's 132 challenges with no miss
    allowed and Δ + RTT = 500 µs (`audit.CHALLENGES`, `DEADLINE_NS + RTT_ALLOWANCE_NS`), and at 4 rounds with one miss
    allowed, small enough to lower whole."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pous.verdict_circuit import PousVerdict

    return [
        bind(PousVerdict, K=132, M=0, LIMIT=500_000),
        bind(PousVerdict, K=4, M=1, LIMIT=500_000),
    ]
