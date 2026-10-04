"""Network accounting: the most the site could have sent out, from what the warden saw and an assumed bound on the side
channels it cannot see.

    verifier    reads the traffic, and records the account
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.primitives.events import Actor, Send, receive
from verity.protocols.transparency import Traffic


@dataclass(frozen=True)
class NetworkAccount:
    """At most `bits` left the site: what the warden saw, its timing, and the side channels (assumed)."""

    bits: int
    accepted: bool


@dataclass(frozen=True)
class Network:
    side_bits: int  # what the side channels carry over the period: an assumption

    def verifier(self) -> Actor:
        traffic = yield from receive(Traffic, "verifier")
        return (Send(NetworkAccount(8 * traffic.bytes + traffic.timing_bits + self.side_bits, traffic.accepted)),)
