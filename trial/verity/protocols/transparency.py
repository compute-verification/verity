"""Network transparency: a warden on the link sees every byte that leaves the site, and shapes the traffic so that its
timing carries at most a known number of bits.

    warden      reports what left over the period and the timing channel's capacity (its readings, signed)
    verifier    audits the report, and records the traffic

Stands in for `verity_network_warden`, whose audit replays the warden's records against the logical schedule; here it
only checks the timing channel against `max_timing_bits`.
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.primitives.events import Actor, Send, receive, tell


@dataclass(frozen=True)
class Observed:
    bytes: int
    timing_bits: int


@dataclass(frozen=True)
class Traffic:
    """If `accepted`: `bytes` left the site, and their timing carried at most `timing_bits`."""

    bytes: int
    timing_bits: int
    accepted: bool


@dataclass(frozen=True)
class Transparency:
    max_timing_bits: int
    name: str = "transparency"

    def verifier(self, to: tuple[str, ...] = ()) -> dict[str, Actor]:
        return {f"verifier/{self.name}": self._verifier(to)}

    def warden(self, bytes_out: int, timing_bits: int) -> dict[str, Actor]:
        return {f"warden/{self.name}": tell(Send(Observed(bytes_out, timing_bits), to=(f"verifier/{self.name}",)))}

    def _verifier(self, to: tuple[str, ...]) -> Actor:
        observed = yield from receive(Observed, "warden")
        return (Send(Traffic(observed.bytes, observed.timing_bits, observed.timing_bits <= self.max_timing_bits), to=to),)
