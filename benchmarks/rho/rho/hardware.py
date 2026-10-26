"""The actual datacenter's hardware: locations, fleets and horizons.

A **location** is the unit the physical cut charges: the largest set of devices joined by links no certifier sees (an
8-GPU server's NVLink domain, an NVL72 rack, or a rack of servers whose certifiers sit only on its uplinks). Its `phi`
is its peak FLOP/s at the training precision, `memory` its devices' bits, and its *hosts* (the servers or compute
trays, each with its own NICs, certifier and side channels) carry host memory, host CPUs and the boundary links whose
line rate is `links` (`β_X`, `Accounting.linkCapacity`). What a wipe leaves in it is the onsite regime's
(`regimes.Onsite`).

A location's size moves the physical cut: an NVL72 rack has 9 times an HGX H200 server's devices, so 9 times its wipe
residual, and 23 times its compute, so 23 times its certified ingress (`α Φ`). Its window's input `K` grows less than
its compute (15 times at today's numbers), so a bound linear in `K` gives it a smaller share and one superlinear in `K`
(`CoverTraining`'s `k²/b`) may give it a larger one; and a state that fits in one location's `K` trains there
unthrottled (Kimi K2's 16-bit state fits a rack's day, not a server's).

A **fleet** is the adversary's monitored locations and its dark share of compute: one connected pool outside every
monitored region, which runs unencumbered (Daniel, 25 Oct), at its own utilization `mfu_dark`. A **horizon** is the
monitoring period a result is over.

A location table may leave out its hosts: it is then one host with no host memory or CPUs counted and no known line
rate (no `links`-form physical cut).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rho.params import Registry


@dataclass(frozen=True)
class Location:
    name: str
    devices: int
    #: peak FLOP/s of the whole location at the training precision (dense)
    phi: float
    #: the devices' memory, bits
    memory: float
    #: servers or compute trays: what certified timing and side channels are counted per
    hosts: int = 1
    #: host memory (and any storage left plugged in), bits
    host_memory: float = 0.0
    #: the host CPUs' peak ops/s at their densest integer matrix rate (what tiers counts host CPUs at)
    host_ops: float = 0.0
    #: `β_X`: bit/s across the location's boundary, its NICs' line rate (`Accounting.linkCapacity`)
    links: float = math.inf


@dataclass(frozen=True)
class Fleet:
    name: str
    location: str
    #: monitored locations
    count: float
    #: share of the adversary's compute (peak FLOP/s) outside every monitored region, as one connected pool
    dark: float
    #: model FLOP utilization an unencumbered trainer gets on the monitored fleet, and the one its served work runs at
    mfu_train: float
    mfu_serve: float
    #: the utilization the dark pool trains at (one connected pool: its interconnect may beat the monitored fleet's);
    #: None: `mfu_train`
    mfu_dark: float | None = None

    @property
    def dark_effective(self) -> float:
        """The dark pool's share of the adversary's unencumbered training progress: `d` weighted by its utilization,
        `d μ_d / (d μ_d + (1 − d) μ_t)`, which is `d` when the pool trains like the monitored fleet."""
        if self.mfu_dark is None:
            return self.dark
        dark = self.dark * self.mfu_dark
        return dark / (dark + (1 - self.dark) * self.mfu_train)


@dataclass(frozen=True)
class Horizon:
    name: str
    seconds: float


def location(registry: Registry, name: str) -> Location:
    return registry.fields(Location, f"location.{name}")


def fleet(registry: Registry, name: str) -> Fleet:
    return registry.fields(Fleet, f"fleet.{name}")


def horizon(registry: Registry, name: str) -> Horizon:
    return registry.fields(Horizon, f"horizon.{name}")
