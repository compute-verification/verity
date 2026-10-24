"""The integration's one refusal type, `Refused`: what was asked is outside what the code serves or proves, said by reason.

Lint P14 (`tests/lint/test_p14_one_way.py`) holds every other refusal class to its shrinking allowlist.
"""
from __future__ import annotations

from collections.abc import Sequence


class Refused(ValueError):
    """``reasons`` lists every reason (one string is one reason).  ``what`` names what refused: the message is
    ``"<what> refused: <reason>; <reason>"``, or the reasons alone without it."""

    def __init__(self, reasons: str | Sequence[str], what: str = "") -> None:
        self.reasons = (reasons,) if isinstance(reasons, str) else tuple(reasons)
        self.what = what
        said = "; ".join(self.reasons)
        super().__init__(f"{what} refused: {said}" if what else said)


__all__ = ["Refused"]
