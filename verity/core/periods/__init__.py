"""Ticks, spans of ticks and the monitoring period: the Python mirror of the code's `CoreProtocol.Period`
(`verity/core/protocols/interface/CoreProtocol/Period.lean`), which is the code's copy of
`Security/Definitions/Core/Period.lean`. The difftest port `core-periods-difftest` holds this to that Lean. Stdlib only,
and it imports no other unit.

- `TickSpan(start, len)`: the ticks ``[start, start + len)`` on the one clock (`OneClock`), ending at `stop`. `ticks` is
  their range, and `within` says one span lies inside another.
- `Cut(span, boundaries)`: a span cut at its boundaries, from its start to its stop, each no later than the next. Its
  `parts` are the spans between consecutive boundaries.
- `MonitoringPeriod(span, execution, sampling)`: a span, its cut into its execution blocks' spans, and for each execution
  block its span's cut into its sampling blocks' spans. `MonitoringPeriod.of` builds one from boundaries.
- `MONITORING_PERIODS`: the monitoring periods a result is presented for (week, month, year), each with its length in
  seconds.

The JSON is the Lean's: a span is ``{"len": n, "start": n}`` and a monitoring period ``{"execution": [τ, …], "sampling":
[[τ, …], …], "span": SPAN}``. `from_json` accepts exactly what `to_json` writes. Anything that isn't a span, a cut or a
monitoring period raises `PeriodError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: The monitoring periods a result is presented for, each with its length in seconds.
MONITORING_PERIODS: tuple[tuple[str, int], ...] = (("week", 7 * 86400), ("month", 30 * 86400), ("year", 365 * 86400))


class PeriodError(ValueError):
    """A span, cut or monitoring period that isn't one, or JSON that encodes none."""


def _nat(v: Any, what: str) -> int:
    if type(v) is not int or v < 0:
        raise PeriodError(f"{what}: not a natural number")
    return v


def _object(j: Any, keys: tuple[str, ...], what: str) -> dict[str, Any]:
    if not isinstance(j, dict):
        raise PeriodError(f"{what}: not an object")
    if sorted(j) != list(keys):
        raise PeriodError(f"{what}: the keys must be {list(keys)}")
    return j


def _ticks(j: Any, what: str) -> tuple[int, ...]:
    if not isinstance(j, list):
        raise PeriodError(f"{what}: not a list")
    return tuple(_nat(t, what) for t in j)


@dataclass(frozen=True)
class TickSpan:
    """A span of ticks: the ticks ``[start, start + len)``."""

    start: int
    len: int

    def __post_init__(self) -> None:
        _nat(self.start, "start")
        _nat(self.len, "len")

    @property
    def stop(self) -> int:
        """The tick after the span's last."""
        return self.start + self.len

    @property
    def ticks(self) -> range:
        return range(self.start, self.stop)

    def within(self, outer: TickSpan) -> bool:
        """Whether this span starts no earlier than `outer` and stops no later."""
        return outer.start <= self.start and self.stop <= outer.stop

    def to_json(self) -> dict[str, int]:
        return {"len": self.len, "start": self.start}

    @classmethod
    def from_json(cls, j: Any) -> TickSpan:
        o = _object(j, ("len", "start"), "span")
        return cls(_nat(o["start"], "span.start"), _nat(o["len"], "span.len"))


@dataclass(frozen=True)
class Cut:
    """The span `span` cut at `boundaries`: the first its start, the last its stop, each no later than the next."""

    span: TickSpan
    boundaries: tuple[int, ...]

    def __post_init__(self) -> None:
        bs = tuple(_nat(b, "boundary") for b in self.boundaries)
        object.__setattr__(self, "boundaries", bs)
        if not bs or bs[0] != self.span.start or bs[-1] != self.span.stop or any(a > b for a, b in zip(bs, bs[1:])):
            raise PeriodError(f"the boundaries {list(bs)} don't cut {self.span}")

    @property
    def parts(self) -> tuple[TickSpan, ...]:
        """The spans between consecutive boundaries."""
        return tuple(TickSpan(a, b - a) for a, b in zip(self.boundaries, self.boundaries[1:]))


@dataclass(frozen=True)
class MonitoringPeriod:
    """A monitoring period: its span, the cut of its span into its execution blocks' spans, and for each execution block
    the cut of its span into its sampling blocks' spans."""

    span: TickSpan
    execution: Cut
    sampling: tuple[Cut, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampling", tuple(self.sampling))
        if self.execution.span != self.span:
            raise PeriodError(f"the execution blocks cut {self.execution.span}, not the span {self.span}")
        if len(self.sampling) != len(self.execution.parts):
            raise PeriodError(f"{len(self.sampling)} sampling cuts for {len(self.execution.parts)} execution blocks")
        for j, (block, cut) in enumerate(zip(self.execution.parts, self.sampling)):
            if cut.span != block:
                raise PeriodError(f"execution block {j}'s sampling cut is of {cut.span}, not of its span {block}")

    @classmethod
    def of(cls, span: TickSpan, execution: Sequence[int], sampling: Sequence[Sequence[int]]) -> MonitoringPeriod:
        """The monitoring period of `span` whose execution blocks' boundaries are `execution`, and whose execution block
        `j`'s sampling blocks' boundaries are `sampling[j]`."""
        cut = Cut(span, tuple(execution))
        if len(sampling) != len(cut.parts):
            raise PeriodError(f"{len(sampling)} sampling cuts for {len(cut.parts)} execution blocks")
        return cls(span, cut, tuple(Cut(block, tuple(bs)) for block, bs in zip(cut.parts, sampling)))

    @property
    def execution_blocks(self) -> int:
        return len(self.sampling)

    def execution_span(self, j: int) -> TickSpan:
        """Execution block `j`'s span."""
        return self.execution.parts[j]

    def sampling_span(self, j: int, t: int) -> TickSpan:
        """Sampling block `t`'s span in execution block `j`."""
        return self.sampling[j].parts[t]

    def to_json(self) -> dict[str, Any]:
        return {"execution": list(self.execution.boundaries), "sampling": [list(c.boundaries) for c in self.sampling],
                "span": self.span.to_json()}

    @classmethod
    def from_json(cls, j: Any) -> MonitoringPeriod:
        o = _object(j, ("execution", "sampling", "span"), "monitoring period")
        span = TickSpan.from_json(o["span"])
        execution = _ticks(o["execution"], "monitoring period.execution")
        if not isinstance(o["sampling"], list):
            raise PeriodError("monitoring period.sampling: not a list")
        return cls.of(span, execution, [_ticks(bs, "monitoring period.sampling") for bs in o["sampling"]])


__all__ = ["MONITORING_PERIODS", "Cut", "MonitoringPeriod", "PeriodError", "TickSpan"]
