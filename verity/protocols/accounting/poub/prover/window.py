"""PoUB's read session's statements, handed to the proof system as any integration hands a window over
(`verity.commitments.periods.interface`): one opening unit (`BlockOpeningCheck_v2`, the verifier's `block_check`) a
sample of the session's window, its values recorded into a `Window`. The verifier decides the window from the proof
system's accepted runs of those units (`RemoteAudit.Span.checkSpan`), opening each unit's ``pub``, ``signs`` and ``o0``
rows (the verifier's `span`).

`SessionIntegration` is a window's statements, an `Integration`: its `circuit()` and its `run(window)`. Neither is
written yet.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verity.commitments.periods.interface import Circuit, Window

GOAL = "memory-accounting/read-session-in-zero-knowledge"


class SessionIntegration:
    """PoUB's read session as an integration: one opening unit per sample (i, p) of the window, in sample order, over
    block j = p of a pass whose plan is the identity. Its ``pub`` and ``signs`` rows are the verifier's own
    (`span.public_rows`); its block, payload, leaf and aux rows are the store's and the pass's, read only inside the
    proof; its output ``o0`` is the verdict word. `samples` is the window's position samples (`audit.PositionSample`)."""

    def __init__(self, samples: Sequence[Any]) -> None:
        self.samples = tuple(samples)

    def circuit(self) -> Circuit:
        raise NotImplementedError(f"{GOAL}: PoUB's read session as a window's circuit (SessionIntegration.circuit)")

    def run(self, window: Window) -> None:
        raise NotImplementedError(f"{GOAL}: PoUB's read session's values recorded into a window (SessionIntegration.run)")
