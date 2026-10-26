"""The actual datacenter's hardware: locations, fleets and horizons.

A **location** is the unit the physical cut charges: the largest set of devices joined by links no certifier sees (an
8-GPU server's NVLink domain, an NVL72 rack). Its `phi` is its peak FLOP/s at the training precision and `memory` its
bits; what a wipe leaves in it is the onsite regime's (`regimes.Onsite`). A **fleet** is the
adversary's monitored locations and its dark share of compute: one connected pool outside every monitored region, which
runs unencumbered (Daniel, 25 Oct). A **horizon** is the monitoring period a result is over.
"""

from __future__ import annotations

from dataclasses import dataclass

from rho.params import Registry


@dataclass(frozen=True)
class Location:
    name: str
    devices: int
    #: peak FLOP/s of the whole location at the training precision (dense)
    phi: float
    #: memory bits
    memory: float


@dataclass(frozen=True)
class Fleet:
    name: str
    location: str
    #: monitored locations
    count: float
    #: share of the adversary's compute outside every monitored region, as one connected pool
    dark: float
    #: model FLOP utilization an unencumbered trainer gets, and the one the declared (served) work runs at
    mfu_train: float
    mfu_serve: float


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
