"""circuit-check's bindings of `verity.core.protocols.pous.answer_circuit`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pous_roots() -> list[Any]:
    """PoUS's per-answer opening (`verity.core.protocols.pous.answer_circuit`) of a 264-byte row reaches the row
    digest's head, a middle compression and a tail with its rest."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pous.answer_circuit import PousAnswerOpens

    return [
        bind(PousAnswerOpens, N=264),
    ]
