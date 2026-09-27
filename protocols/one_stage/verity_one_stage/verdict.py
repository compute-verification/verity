from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Verdict:
    """A check's outcome: `ok`, the failure `codes` in the order found, and a `detail` per code."""

    ok: bool
    codes: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "codes": list(self.codes), "detail": self.detail}


class _Collect:
    def __init__(self) -> None:
        self.detail: dict[str, Any] = {}

    def fail(self, code: str, what: Any) -> None:
        self.detail.setdefault(code, what)

    def verdict(self) -> Verdict:
        return Verdict(not self.detail, tuple(self.detail), dict(self.detail))
