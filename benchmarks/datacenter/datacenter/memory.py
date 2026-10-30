"""Stores, challenges, read audits and wipes: how timed audits bind state to devices.

A `StoreEntry` is registered state: `bits` bits the developer says a device holds. Its `content` says who chose them:
the prover (a KV cache, which may itself hide state) or the verifier (`labels`: the challenger's labels, which hide
nothing and so displace whatever else the device held).

The trusted challenger asks for a sample of a store at a tick and wants the answer by a deadline (`Challenge`, PoUS), or
has the prover read a store through a window and counts the bits it read (`ReadWindow`, the bandwidth audits). A
`Wipe` is a store of labels the challenger fills a device's memory with, spot-checked within its interval: its `bits`
are a protocol parameter, chosen by the verifier, not read off the device.

Where the bits really are decides how fast they can be served (`read_rate`, `serve_time`): the holding device's read
bandwidth, and, when they sit on another device, the path from there (each link's latency, the bits at its narrowest
link). A deadline that only the registered holder can meet turns "the answer was on time" into "the bits were on that
device": by latency for a small sample, by bandwidth for a whole read. Labels the prover could rebuild from a seed
faster than the deadline locate nowhere. A wipe fills a device no faster than its write ceiling, else the links into
it (`fill_rate`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from verity.core.protocols.remote_bandwidth.codec import BLOCK_BYTES

from .physical import Datacenter
from .time import Interval


@dataclass(frozen=True)
class StoreEntry:
    """Registered state: `bits` bits that `device` holds, by the registration. `content` says who chose them. `regen` is
    what rebuilding one bit costs, in the codec's squarings (`rate["square-1024"]`), for labels the prover can derive
    from a seed (0: it can't, as for its own data or labels the challenger streams in)."""

    name: str
    bits: float
    device: str
    content: Literal["prover", "labels"] = "prover"
    regen: float = 0.0


@dataclass(frozen=True)
class Challenge:
    """`challenger` asks for `sample` bits of store `store` (PoUS's s sampled blocks; None: all of it) at tick `at`
    and wants the answer within `deadline` ticks."""

    store: str
    at: int
    deadline: int
    challenger: str = "challenger"
    sample: float | None = None


@dataclass(frozen=True)
class ReadWindow:
    """One window of a bandwidth audit (the remote audit's window t): `challenger` sends its challenge at `start`, the
    prover reads store `store` and commits receipts until `boundary`, and its roots are due by `deadline` (the boundary
    plus the allowance A), before an openings coin checks them. How many bits were read is the window's output."""

    store: str
    start: int
    boundary: int
    deadline: int
    challenger: str = "challenger"

    @property
    def during(self) -> Interval:
        return Interval(self.start, self.boundary)


@dataclass(frozen=True)
class Wipe:
    """A verified wipe: `challenger` fills `device` with `bits` of labels and spot-checks `sample` bits of them (None:
    all) within `during`. `regen` is what rebuilding one label bit costs the prover in squarings (0 if the challenger
    streams the labels in)."""

    device: str
    bits: float
    during: Interval
    challenger: str = "challenger"
    sample: float | None = None
    regen: float = 0.0

    @property
    def store(self) -> StoreEntry:
        return StoreEntry(f"labels@{self.device}", self.bits, self.device, "labels", self.regen)


BLOCK_BITS = BLOCK_BYTES * 8


@dataclass(frozen=True)
class Append:
    """Blocks `first` … `first + count − 1` appended to store `store` (KV pages): their digests arrive at `at`, the
    verifier releases the append coin at `coin_at`, and they count toward the store from `resident_at`, when their
    encode lands (deferred residency)."""

    store: str
    at: int
    first: int
    count: int
    coin_at: int
    resident_at: int


@dataclass(frozen=True)
class Release:
    """Blocks `first` … `first + count − 1` of store `store` released at `at`: no later read or plan touches them. A
    block released before it was resident is withdrawn, and never counts."""

    store: str
    first: int
    count: int
    at: int


@dataclass(frozen=True)
class Rekey:
    """Blocks `old_first` … re-keyed at their lifetime cap: the old indices are released at `at`, and the same count
    at `new_first` … counts from `resident_at`."""

    store: str
    old_first: int
    new_first: int
    count: int
    at: int
    resident_at: int


def read_rate(dc: Datacenter, registered: str, actual: str) -> float:
    """Bits a second a store registered on `registered` can be read at from where it really is, `actual`: the holder's
    read bandwidth, and from another device no more than the path's narrowest link."""
    from .binding import route
    holder = dc.devices[actual]
    rate = float(holder.bandwidth) if holder.bandwidth else float("inf")
    if actual != registered:
        rate = min(rate, min(float(l.capacity) for l in route(dc, actual, registered)))
    return rate


def serve_time(dc: Datacenter, registered: str, actual: str, bits: float) -> float:
    """Seconds to serve `bits` of a store registered on `registered` from where it really is, `actual`: the read time at
    `read_rate`, plus the path's latency from another device."""
    from .binding import route
    latency = sum(l.delay for l in route(dc, actual, registered)) if actual != registered else 0.0
    return bits / read_rate(dc, registered, actual) + latency


def answer_time(dc: Datacenter, challenger: str, registered: str, actual: str, bits: float) -> float:
    """Seconds from a challenge to its answer: the challenger's round trip to the registered device, plus serving."""
    from .binding import route
    return 2 * sum(l.delay for l in route(dc, challenger, registered)) + serve_time(dc, registered, actual, bits)


def regen_time(dc: Datacenter, store: StoreEntry, bits: float) -> float:
    """Seconds the prover needs to rebuild `bits` of `store` from its seed at its block's summed `rate["square-1024"]`,
    as `ledger.Node.rebuild` sums a node's; infinite for a store that can't be rebuilt."""
    block = dc.block_of(store.device)
    devices = block.devices if block is not None else dc.devices.values()
    rate = sum(float(d.rate["square-1024"]) for d in devices if "square-1024" in d.rate and not d.trusted)
    return bits * store.regen / rate if store.regen and rate else float("inf")


def into(dc: Datacenter, device: str) -> float:
    """Bits a second every path into `device` from the rest of the datacenter carries: the links that end at it."""
    return sum(float(l.capacity) for l in dc.links if device in (l.a, l.b))


def fill_rate(dc: Datacenter, device: str) -> float:
    """Bits a second a wipe can write into `device`: its write ceiling, else the links into it."""
    d = dc.devices[device]
    return float(d.write) if d.write is not None else into(dc, device)
