"""Stores and challenges: how a timed audit binds held bits to a device.

A `Store` is registered state: `bits` bits the developer says a device holds. The trusted challenger asks for a store at a
tick and wants the answer by a deadline (`Challenge`). Where the bits really are decides when the answer can arrive:
the holding device's read time (its `bandwidth`), plus, when they sit on another device, the path there and back (each
link's latency, and the bits at the path's narrowest capacity). A deadline that the honest holder meets and every other
holder misses turns "the answer was on time" into "the bits were on that device".

This is why the trusted clock carries real durations: the memory audit reads the length of the interval between the
challenge and the deadline, not just the order of events.
"""
from __future__ import annotations

from dataclasses import dataclass

from .binding import route
from .physical import Datacenter
from .time import Clock


@dataclass(frozen=True)
class Store:
    """Registered state: `bits` bits that `device` holds, by the developer's registration."""

    name: str
    bits: float
    device: str


@dataclass(frozen=True)
class Challenge:
    """The challenger asks `challenger` -> holder for store `store` at tick `at`, and wants the answer within `deadline`
    ticks."""

    store: str
    at: int
    deadline: int
    challenger: str = "challenger"


def answer_time(dc: Datacenter, clock: Clock, challenger: str, registered: str, actual: str, bits: float,
                sample: float = 1.0) -> float:
    """Seconds from the challenge to its answer when the bits registered on `registered` actually sit on `actual`: the
    round trip from the challenger to the registered device, the read on `actual`, and, if `actual` is another device,
    the path from it to the registered device carrying `sample`·bits."""
    there = route(dc, challenger, registered)
    seconds = 2 * sum(l.latency for l in there)
    holder = dc.devices[actual]
    read = sample * bits / float(holder.bandwidth) if holder.bandwidth else 0.0
    seconds += read
    if actual != registered:
        path = route(dc, actual, registered)
        seconds += sum(l.latency for l in path) + sample * bits / min(float(l.capacity) for l in path)
    return seconds
