"""Canonical GPU and machine names; naming systems remain distinct fields."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

from .loader import ConfigError, load

REGISTRY = Path(__file__).with_name("registry")


@dataclass(frozen=True)
class Gpu:
    id: str
    census_id: str
    nvidia_smi_name: str
    nebius_platform: str
    nebius_preset: str
    runpod_type_id: str
    memory_bytes: int
    spec_bandwidth_bytes_per_second: int
    count_per_node: int


@dataclass(frozen=True)
class GpuRegistry:
    schema: str
    gpu: tuple[Gpu, ...]


@dataclass(frozen=True)
class Machine:
    id: str
    research_name: str
    provider: str
    platform: str
    preset: str
    gpu_id: str | None
    gpu_count: int


@dataclass(frozen=True)
class MachineRegistry:
    schema: str
    machine: tuple[Machine, ...]


def _unique(items: tuple[Gpu, ...] | tuple[Machine, ...], kind: str) -> None:
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ConfigError(f"{kind} registry has duplicate ids")


@cache
def gpus(path: str | Path = REGISTRY / "gpus.toml") -> dict[str, Gpu]:
    doc = load(path, GpuRegistry)
    if doc.schema != "verity-config/gpus/v0":
        raise ConfigError(f"{path}: unknown GPU registry schema {doc.schema!r}")
    _unique(doc.gpu, "GPU")
    return {gpu.id: gpu for gpu in doc.gpu}


@cache
def machines(path: str | Path = REGISTRY / "machines.toml") -> dict[str, Machine]:
    doc = load(path, MachineRegistry)
    if doc.schema != "verity-config/machines/v0":
        raise ConfigError(f"{path}: unknown machine registry schema {doc.schema!r}")
    _unique(doc.machine, "machine")
    known = gpus()
    bad = sorted({machine.gpu_id for machine in doc.machine if machine.gpu_id is not None} - set(known))
    if bad:
        raise ConfigError(f"{path}: machines name unknown GPU ids: {bad}")
    return {machine.id: machine for machine in doc.machine}
