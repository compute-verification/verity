"""circuit-check's bindings of `verity.core.protocols.certifier.advice_check`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_advice_roots() -> list[Any]:
    """Advice v0's validity check (`verity.core.protocols.certifier.advice_check`) at L = 40, the least cap, which
    composes every part at its smallest in about 20 s; and its five parts at L = 240, which holds the tests' fixture,
    each within the whole-Definition lowering budget the units exceed, so each part's Flock lowering is checked and
    pinned. The unit at L = 240 takes about 150 s, so it stays out of `check`."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.advice_check import PARTS, unit

    return [unit(40), *(bind(d, L=240) for d in PARTS)]
