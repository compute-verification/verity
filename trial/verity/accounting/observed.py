"""Observed: how much of a resource the verifier's hardware at the site saw used, by whatever used it. Today only the
network: a warden on every link out reports the bytes that left over the epoch and the bits their timing could carry,
and the verifier audits the report. What no warden sees, the side channels, is an assumption (`side_bits`).

    warden      on every link out of the site: reports what left (its readings)
    verifier    audits the warden's report, and records the observation

Stands in for `verity_network_warden`, whose warden shapes frames onto a grid of time slots and whose audit replays its
records against the logical schedule; here the audit only checks the timing channel against `max_timing_bits`.
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.primitives.events import Actor, Send, receive, tell


@dataclass(frozen=True)
class Left:
    """Warden to verifier: the bytes that left over the epoch, and the bits their timing could carry."""

    bytes: int
    timing_bits: int


@dataclass(frozen=True)
class Observed:
    """If no `reasons`: `seen` of `resource` was used, and at most `unseen` more where nothing at the site looks."""

    resource: str
    seen: int
    unseen: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Network:
    max_timing_bits: int
    side_bits: int  # what the side channels carry over an epoch: an assumption

    def verifier(self) -> Actor:
        left = yield from receive(Left, "warden")
        reasons = (
            (f"a timing channel of {left.timing_bits} bits, over {self.max_timing_bits}",)
            if left.timing_bits > self.max_timing_bits
            else ()
        )
        return (Send(Observed("network", 8 * left.bytes, left.timing_bits + self.side_bits, reasons)),)

    def warden(self, bytes_out: int, timing_bits: int) -> Actor:
        return tell(Send(Left(bytes_out, timing_bits), "verifier"))
