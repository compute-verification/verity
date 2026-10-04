"""Capacity: what the site could have supplied of each resource, from a physical description of it. The `Site` is the
devices at it by type, as whoever the verifier trusts reports them; the census gives each type's facts (`Device`), which
are assumptions. Remotely the prover reports the site, so capacity holds only if it left no device out; onsite the
pinger counts it, the most devices at the site at any time in the window.

    surveyor    the prover (remote) or the pinger (onsite): reports the site
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.primitives.events import Actor, Send, tell

RESOURCES = ("compute", "memory", "network")


@dataclass(frozen=True)
class Device:
    """A device type's facts, the census's."""

    work: float = 0.0  # work a second, at peak
    memory: int = 0  # bytes, volatile
    egress: float = 0.0  # bits a second it can send out of the site


@dataclass(frozen=True)
class Site:
    devices: Mapping[str, int]  # device type -> how many


def capacity(site: Site, census: Mapping[str, Device], seconds: float) -> dict[str, int]:
    """Each resource's capacity over `seconds`: compute and network are rates, memory is held."""
    facts = [(n, census.get(device, Device())) for device, n in site.devices.items()]
    return {
        "compute": int(seconds * sum(n * d.work for n, d in facts)),
        "memory": sum(n * d.memory for n, d in facts),
        "network": int(seconds * sum(n * d.egress for n, d in facts)),
    }


def survey(devices: Mapping[str, int]) -> Actor:
    return tell(Send(Site(devices), "verifier"))
