"""circuit-check's bindings of `verity.core.protocols.pous.circuit`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pous_roots() -> list[Any]:
    """P2's modular squaring (`verity.core.protocols.pous.circuit`) at W = 128 with P2's C0 = 21,065: its square
    takes a Toom-3 level (above `TOOM`), then Karatsuba (above `SCHOOL`), then schoolbook, so it reaches every piece; the
    package's tests check it against Python's integers at P2's width and pin its gate counts there."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pous.circuit import PousModSquare

    return [
        bind(PousModSquare, W=128, C0=21065),
    ]
