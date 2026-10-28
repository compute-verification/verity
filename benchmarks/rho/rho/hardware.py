"""The physical model, `𝒜_phys`: the inventory's locations, its dark leaves and the horizon, in gates (the working
draft's §8).

`Units` is the one place a FLOP-model input becomes gates: `units.gates_per_flop` (`data/hardware.toml`), with its source
and range. The datasheets' FLOP/s, the policy's `Γ`, the run's target, a sampled step's FLOPs and the classes' and
attacks' FLOP-model statements all pass through it, so under `structure.reference` (whose `s_×` moves with it) the
overhead and the time to the target don't depend on its value (`tests/test_hardware.py`). Its `basis` says where the
value comes from (`FLOP_MODEL`, or `boolean:<design>` when measured on one Boolean design); a structure's `s_×` must
come from the same one (`classes.Structure.check`).

A **location** is the largest set of devices joined by links no certifier sees (an 8-GPU server's NVLink domain, an NVL72
rack, or a rack of servers whose certifiers sit only on its uplinks). Its `phi` is `Φ_X`, its capacity in gates per
second (`|A[X,J]| ≤ Φ_X |J|`), `memory` its devices' bits, and its *hosts* (the servers or compute trays, each with its
own NICs, certifier and side channels) carry host memory, disks, host CPUs and the boundary links whose line rate is
`links` (`β_X`, `Accounting.linkCapacity`). `server_memory` is `M_s`, one server's memory over every tier (HBM, host
RAM, disks): the capacity the wipe's per-server bound is stated against.

A **fleet** is the inventory: the monitored locations, and the dark leaves `L_dark` with `Φ(L_dark) = ε_dark Φ_total`,
one connected pool outside every monitored location (ruled). The physical model bounds every gate at capacity,
so the bound counts the dark leaves at `Φ`; `mfu_train` and `mfu_dark` are what a construction (an attack) gets of the
monitored and the dark gates. A **horizon** is the monitoring period a result is over.

A location table may leave out its hosts: it is then one host with no host memory or CPUs counted and no known line
rate (no `links`-form serving block).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rho.params import Registry

#: `gates_per_flop` as an assumed conversion of the FLOP model's inputs, not measured on a Boolean design
FLOP_MODEL = "flop-model"


@dataclass(frozen=True)
class Units:
    """The physical model's one conversion from the FLOP model to Boolean gates."""

    #: Boolean gates per FLOP at the training precision
    gates_per_flop: float
    #: where `gates_per_flop` comes from: `FLOP_MODEL`, or `boolean:<design>` for a count on one Boolean design
    basis: str = FLOP_MODEL

    def gates(self, flops: float) -> float:
        return flops * self.gates_per_flop

    def flops(self, gates: float) -> float:
        return gates / self.gates_per_flop


@dataclass(frozen=True)
class Location:
    name: str
    devices: int
    #: `Φ_X`: gates per second of the whole location (its dense FLOP/s at the training precision, through `Units`)
    phi: float
    #: the devices' memory, bits
    memory: float
    #: servers or compute trays: what the wipe's residual, certified timing and side channels are counted per
    hosts: int = 1
    #: host memory, bits
    host_memory: float = 0.0
    #: disks left plugged in, bits
    disk: float = 0.0
    #: the host CPUs in gates per second (their densest integer matrix rate, through `Units`)
    host_phi: float = 0.0
    #: `β_X`: bit/s across the location's boundary, its NICs' line rate (`Accounting.linkCapacity`)
    links: float = math.inf

    @property
    def server_memory(self) -> float:
        """`M_s`: one server's memory over every tier, bits."""
        return (self.memory + self.host_memory + self.disk) / self.hosts


@dataclass(frozen=True)
class Fleet:
    name: str
    location: str
    #: monitored locations
    count: float
    #: `ε_dark`: the dark leaves' share of the inventory's capacity, `Φ(L_dark) / Φ_total`
    eps_dark: float
    #: the share of a monitored location's gates a construction trains with, and the one its served work runs at
    mfu_train: float
    mfu_serve: float
    #: the share of the dark leaves' gates a construction trains with (one connected pool: its interconnect may beat
    #: the monitored fleet's); None: `mfu_train`
    mfu_dark: float | None = None

    @property
    def dark_mfu(self) -> float:
        return self.mfu_train if self.mfu_dark is None else self.mfu_dark


@dataclass(frozen=True)
class Horizon:
    name: str
    seconds: float


def units(registry: Registry) -> Units:
    return registry.fields(Units, "units")


def location(registry: Registry, name: str) -> Location:
    """Location `name`, its datasheet rates (`flops`, `host_ops`) in gates through `Units`."""
    u = units(registry)
    t = f"location.{name}"
    return registry.fields(Location, t, skip=("flops", "host_ops"), phi=u.gates(registry[f"{t}.flops"]),
                           host_phi=u.gates(registry.get(f"{t}.host_ops", 0.0)))


def fleet(registry: Registry, name: str) -> Fleet:
    return registry.fields(Fleet, f"fleet.{name}")


def horizon(registry: Registry, name: str) -> Horizon:
    return registry.fields(Horizon, f"horizon.{name}")
