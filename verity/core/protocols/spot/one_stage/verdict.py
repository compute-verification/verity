from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Verdict:
    """A check's outcome: `ok`, the failure `codes` in the order found, and a `detail` per code. `verifiers`, when the
    check ran through `verifier`, is both verifiers' outcomes and which one is of record. `fanout`, what a registration
    check publishes under a template query: the fan-out F it computed on its own program (R8, `partition.fanout`).
    `tied`, what it publishes beside F: the tied parameters whose run Input R8 let through though several proof blocks
    read it, so that its acceptance rests on the later check that ties their rows (`registration.check`'s `tied`)."""

    ok: bool
    codes: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)
    verifiers: dict[str, Any] | None = field(default=None, compare=False)
    fanout: int | None = field(default=None, compare=False)
    tied: tuple[str, ...] = field(default=(), compare=False)

    def to_json(self) -> dict[str, Any]:
        out = {"ok": self.ok, "codes": list(self.codes), "detail": self.detail}
        if self.fanout is not None:
            out["fanout"] = self.fanout
        if self.tied:
            out["tied"] = list(self.tied)
        return out if self.verifiers is None else {**out, "verifiers": self.verifiers}


class _Collect:
    def __init__(self) -> None:
        self.detail: dict[str, Any] = {}

    def fail(self, code: str, what: Any) -> None:
        self.detail.setdefault(code, what)

    def verdict(self, fanout: int | None = None, tied: tuple[str, ...] = ()) -> Verdict:
        return Verdict(not self.detail, tuple(self.detail), dict(self.detail), fanout=fanout, tied=tied)
