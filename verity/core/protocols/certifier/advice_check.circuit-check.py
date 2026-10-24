"""circuit-check's bindings of `verity.core.protocols.certifier.advice_check`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_advice_roots() -> list[Any]:
    """Advice v0's validity check (`verity.core.protocols.certifier.advice_check`) at L = 40, the least cap (the empty
    advice's five bytes), and at L = 240, which holds the tests' fixture: both reach every part (the parser, its end, the
    sort's swap, the replay and its end)."""
    from verity.core.protocols.certifier.advice_check import unit

    return [unit(40), unit(240)]
