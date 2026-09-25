"""The census: a knowledge base of the fixed quantities benchmarks are computed against, each entry with a stable id, its
sources and, where it has one, its methodology.  The data are the JSON files under ``data/``; this module is their typed
loader.

* :data:`HARDWARE` -- the GPU lines we measure: a device and an operand dtype (the instruction registry's ``a_dtype``:
  ``bf16``, ``e4m3``, ``e2m1``) with the datasheet dense tensor-core peak at FP32 accumulate.  :func:`line` finds one by
  device and dtype.
* :data:`NETWORKS` -- :data:`REFERENCE_NETWORK` (same datacenter, 1 ms RTT, 100 Gb/s), at which a Table 2 cell is evaluated,
  and :data:`SENSITIVITY_NETWORKS`, the interaction drill-down's other profiles.
* :data:`WORKLOADS` -- the unit a table row proves (``GemmCoordinate<K>`` at the frozen K and reference batch).
* :data:`INSTANCE_SETS` -- the frozen instance set of each target (by target name): dataset, tier, size and digest.

Ids are stable: renders cite them and sites build history over them.  A changed quantity is a new entry with a new id.
The census imports nothing from Verity (``tests/test_census_registry.py``); core never imports the census -- the renderer,
benchmarks and tools read it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

__all__ = [
    "HARDWARE",
    "INSTANCE_SETS",
    "NETWORKS",
    "REFERENCE_NETWORK",
    "SENSITIVITY_NETWORKS",
    "VU_K1536",
    "WORKLOADS",
    "Hardware",
    "InstanceSet",
    "Network",
    "Source",
    "Workload",
    "line",
]


@dataclass(frozen=True, slots=True)
class Source:
    """Where an entry's value comes from: a ``label`` (document, table, row) and optionally an ``href``."""

    label: str
    href: str | None = None


@dataclass(frozen=True, slots=True)
class Hardware:
    """One measured GPU line: ``device`` (the datasheet part) running operand ``dtype``, with its dense tensor-core peak at
    the ``accumulate`` width in FLOP per second.  ``memory_bytes`` is the reference part's device memory (a board reporting
    a very different total is a variant of the SKU, and this peak is not its denominator); ``None`` = unchecked."""

    id: str
    device: str
    dtype: str
    rate: str
    flop_per_second: float
    accumulate: str
    memory_bytes: int | None
    sources: tuple[Source, ...]
    methodology: str

    @property
    def source(self) -> str:
        return "; ".join(s.label for s in self.sources)

    def seconds_per_vu(self, K: int) -> float:
        """Native wall seconds of one ``GemmCoordinate<K>`` (``2K`` FLOP) at this peak."""

        if not isinstance(K, int) or isinstance(K, bool) or K <= 0:
            raise ValueError(f"K must be a positive int, got {K!r}")
        return 2 * K / self.flop_per_second


@dataclass(frozen=True, slots=True)
class Network:
    """A network profile the interaction formula ``t = compute + rounds x RTT + bytes / bandwidth`` is evaluated at;
    ``role`` is ``reference`` or ``sensitivity``."""

    id: str
    label: str
    rtt_s: float
    bandwidth_bps: float
    role: str
    sources: tuple[Source, ...]


@dataclass(frozen=True, slots=True)
class Workload:
    """The unit a table row proves: ``GemmCoordinate<K>`` measured at the reference ``batch`` of the frozen tables."""

    id: str
    K: int
    batch: int
    description: str
    sources: tuple[Source, ...]


@dataclass(frozen=True, slots=True)
class InstanceSet:
    """A frozen instance set: ``dataset`` / ``tier`` with ``size`` instances, identified by ``manifest_sha256`` (of a
    fixture's ``manifest.json``, or of a synthetic set's recipe string at ``size``); ``seed`` is a synthetic set's generator
    seed (``None`` for a fixture); ``target`` names the target it is the set of."""

    id: str
    target: str
    dataset: str
    tier: str
    size: int
    manifest_sha256: str
    seed: int | None
    sources: tuple[Source, ...]


def _load(name: str) -> list[dict[str, Any]]:
    rows = json.loads(resources.files(__package__).joinpath("data", f"{name}.json").read_text(encoding="utf-8"))
    for r in rows:
        r["sources"] = tuple(Source(**s) for s in r["sources"])
    return rows


def _registry(cls: type, name: str) -> dict[str, Any]:
    entries = [cls(**r) for r in _load(name)]
    out = {e.id: e for e in entries}
    if len(out) != len(entries):
        raise ValueError(f"census data/{name}.json: duplicate ids")
    return out


HARDWARE: dict[str, Hardware] = _registry(Hardware, "hardware")
NETWORKS: dict[str, Network] = _registry(Network, "networks")
WORKLOADS: dict[str, Workload] = _registry(Workload, "workloads")
INSTANCE_SETS: dict[str, InstanceSet] = _registry(InstanceSet, "instance_sets")

REFERENCE_NETWORK: Network = next(n for n in NETWORKS.values() if n.role == "reference")
SENSITIVITY_NETWORKS: tuple[Network, ...] = tuple(n for n in NETWORKS.values() if n.role == "sensitivity")
VU_K1536: Workload = WORKLOADS["vu-k1536"]


def line(device: str, dtype: str) -> Hardware:
    """The hardware line of ``device`` running ``dtype``; ``KeyError`` when the census has none."""
    for h in HARDWARE.values():
        if h.device == device and h.dtype == dtype:
            return h
    raise KeyError(f"no census hardware line for {device!r} / {dtype!r}")
