"""circuit-check's bindings of `verity.experimental.protocols.pous.answer_circuit`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pous_roots() -> list[Any]:
    """PoUS's per-answer opening (`verity.experimental.protocols.pous.answer_circuit`) of a 264-byte row reaches the row
    digest's head, a middle compression and a tail with its rest; the proof service's unit for PoUS is P2's 2,056-byte block,
    unchanged."""
    from verity.ml.circuits.defs import bind
    from verity.experimental.protocols.pous.answer_circuit import PousAnswerOpens, PousAnswerRow

    return [
        bind(PousAnswerOpens, N=264),
        bind(PousAnswerRow, N=2056),
    ]
