"""Canonical GPU names; naming systems remain distinct fields.  A card's figures (memory, bandwidth, SMs, clocks) are
the card files', not this registry's."""

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
    count_per_node: int


@dataclass(frozen=True)
class GpuRegistry:
    schema: str
    gpu: tuple[Gpu, ...]


def _unique(items: tuple[Gpu, ...], kind: str) -> None:
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

