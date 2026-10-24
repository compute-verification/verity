"""circuit-check's bindings of `verity.core.protocols.certifier.advice_check`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_advice_roots() -> list[Any]:
    """Advice v0's validity check (`verity.core.protocols.certifier.advice_check`) at L = 40, the least cap (the empty
    advice's five bytes), and at L = 240, which holds the tests' fixture; and its five parts at L = 240, each within the
    whole-Definition lowering budget that both units exceed, so each part's C-Flock lowering is checked and pinned."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.advice_check import PARTS, unit

    return [unit(40), unit(240), *(bind(d, L=240) for d in PARTS)]
