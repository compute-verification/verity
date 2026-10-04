"""Communication: bits that cross the site's boundary, through any channel. What a warden on every link out saw leave,
and what integrity's checked outputs explain of it, against what the site's links could carry.

    warden      on every link out of the site: reports what left over a window, by its clock
    verifier    audits the warden's report, reads integrity's profile and the site `site_from` reported; records
                communication's line

The stream explains the output bits integrity checked, which leave as responses, up to what the warden saw leave. Every
output counts as a response, so outputs that stay as state (a KV cache) explain as many bits of anything else.
Physically, what no warden sees (the side channels: power, timing on shared links, a radio) is an assumption,
`side_bits` over an epoch, observed beside what it saw; and capacity is the site's devices' egress (the census's) over
the warden's window.

Stands in for `verity_network_warden`, whose warden shapes frames onto a grid of time slots and whose audit replays its
records against the logical schedule; here the audit only checks the timing channel against `max_timing_bits`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.primitives.events import Actor, Send, gather, tell
from verity.primitives.physical import Device, Site, Window
from verity.protocols.accounting.account import line
from verity.protocols.accounting.integrity import IntegrityProfile


@dataclass(frozen=True)
class Left:
    """Warden to verifier: the bytes that left over `window`, and the bits their timing could carry."""

    bytes: int
    timing_bits: int
    window: Window


@dataclass(frozen=True)
class Communication:
    census: Mapping[str, Device]  # device type -> its facts
    max_timing_bits: int
    side_bits: int  # what the side channels carry over an epoch: an assumption

    def verifier(self, site_from: str) -> Actor:
        got = yield from gather({Left: "warden", IntegrityProfile: "verifier", Site: site_from})
        left, profile = got[Left], got[IntegrityProfile]
        egress, reasons = got[Site].supply(self.census, "egress")
        if left.timing_bits > self.max_timing_bits:
            reasons += (f"a timing channel of {left.timing_bits} bits, over {self.max_timing_bits}",)
        if not profile.accepted:
            reasons += ("integrity not established",)
        seen, outputs = 8 * left.bytes, profile.outputs
        observed = seen + left.timing_bits + self.side_bits
        explained = min(outputs.total - outputs.wrong, seen)
        capacity = int(left.window.seconds * egress)
        bits = line(
            "communication", left.window, capacity, outputs.total, observed, explained, seen, profile.delta, reasons
        )
        return (Send(bits),)

    def warden(self, bytes_out: int, timing_bits: int, window: Window) -> Actor:
        return tell(Send(Left(bytes_out, timing_bits, window), "verifier"))
