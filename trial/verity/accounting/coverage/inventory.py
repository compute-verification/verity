"""The inventory: the devices at the site, as whoever the verifier trusts for it reports them. Remotely that is the
prover, so a device it leaves out is one the account never sees: remote coverage holds only under the assumption
that the inventory is complete. Onsite it is the verifier's own count: the most devices at the site at any time in the
work's window, since capacity is the window's seconds times the fleet's peak."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.primitives.events import Actor, Send, tell


@dataclass(frozen=True)
class Fleet:
    devices: Mapping[str, int]  # device type -> how many


def inventory(devices: Mapping[str, int]) -> Actor:
    return tell(Send(Fleet(devices), "verifier"))
