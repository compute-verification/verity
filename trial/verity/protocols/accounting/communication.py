"""Communication: what a warden on every link out of the site saw leave, audited.

    warden      on every link out of the site: reports what left over a window, by its clock
    verifier    audits the warden's report; records `Left`

What no warden sees (the side channels: power, timing on shared links, a radio) and what the checked outputs explain of
what it saw are `reconcile`'s.

Stands in for `verity_network_warden`, whose warden shapes frames onto a grid of time slots and whose audit replays its
records against the logical schedule; here the audit only checks the timing channel against `max_timing_bits`.
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.primitives.events import Actor, Send, receive, tell
from verity.primitives.physical import Window


@dataclass(frozen=True)
class Report:
    """Warden to verifier: the bytes that left over `window`, by its clock, and the bits their timing could carry."""

    bytes: int
    timing_bits: int
    window: Window


@dataclass(frozen=True)
class Left:
    """`bytes` left the site over `window` through the links, and their timing carried at most `timing_bits` bits;
    unless the audit found `reasons`."""

    bytes: int
    timing_bits: int
    window: Window
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Communication:
    max_timing_bits: int

    def verifier(self) -> Actor:
        report = yield from receive(Report, "warden")
        reasons = ()
        if report.timing_bits > self.max_timing_bits:
            reasons = (f"a timing channel of {report.timing_bits} bits, over {self.max_timing_bits}",)
        return (Send(Left(report.bytes, report.timing_bits, report.window, reasons)),)

    def warden(self, bytes_out: int, timing_bits: int, window: Window) -> Actor:
        return tell(Send(Report(bytes_out, timing_bits, window), "verifier"))
