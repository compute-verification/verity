"""Stores, challenges, read audits and wipes: how timed audits bind state to devices.

A `Store` is registered state: `bits` bits the developer says a device holds. Its `content` says who chose them: the
prover (a KV cache, which may itself hide state) or the verifier (`labels`: the challenger's labels, which hide nothing
and so displace whatever else the device held).

The trusted challenger asks for a store at a tick and wants the answer by a deadline (`Challenge`), or wants `bits` of
it read within an interval (`Reads`). A `Wipe` is a store of labels the challenger fills a device's memory with,
spot-checked within its interval: its `bits` are a protocol parameter, chosen by the verifier, not read off the device.

Where the bits really are decides how fast they can be served (`serve_time`): the holding device's read bandwidth, and,
when they sit on another device, the path from there (each link's latency, the bits at its narrowest link). A deadline
that only the registered holder can meet turns "the answer was on time" into "the bits were on that device".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .physical import Datacenter
from .time import Interval


@dataclass(frozen=True)
class Store:
    """Registered state: `bits` bits that `device` holds, by the registration. `content` says who chose them."""

    name: str
    bits: float
    device: str
    content: Literal["prover", "labels"] = "prover"


@dataclass(frozen=True)
class Challenge:
    """`challenger` asks for store `store` at tick `at` and wants the answer within `deadline` ticks."""

    store: str
    at: int
    deadline: int
    challenger: str = "challenger"


@dataclass(frozen=True)
class Reads:
    """A bandwidth audit's demand: `bits` of store `store` read within `during`."""

    store: str
    bits: float
    during: Interval
    challenger: str = "challenger"


@dataclass(frozen=True)
class Wipe:
    """A verified wipe: `challenger` fills `device` with `bits` of labels and spot-checks them within `during`."""

    device: str
    bits: float
    during: Interval
    challenger: str = "challenger"

    @property
    def store(self) -> Store:
        return Store(f"labels@{self.device}", self.bits, self.device, "labels")


def serve_time(dc: Datacenter, registered: str, actual: str, bits: float) -> float:
    """Seconds to serve `bits` of a store registered on `registered` from where it really is, `actual`: the holder's
    read time, or, from another device, the slower of that and the path's narrowest link, plus the path's latency."""
    from .binding import route
    holder = dc.devices[actual]
    seconds = bits / float(holder.bandwidth) if holder.bandwidth else 0.0
    if actual != registered:
        path = route(dc, actual, registered)
        seconds = max(seconds, bits / min(float(l.capacity) for l in path)) + sum(l.delay for l in path)
    return seconds


def answer_time(dc: Datacenter, challenger: str, registered: str, actual: str, bits: float) -> float:
    """Seconds from a challenge to its answer: the challenger's round trip to the registered device, plus serving."""
    from .binding import route
    return 2 * sum(l.delay for l in route(dc, challenger, registered)) + serve_time(dc, registered, actual, bits)
