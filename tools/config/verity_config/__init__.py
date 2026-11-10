"""Strict shared configuration for Verity tools and workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from .canonical import CanonicalError, dumps, sha256, value
from .loader import ConfigError, from_mapping, load
from .registry import Gpu, gpus


@dataclass(frozen=True)
class Admission:
    """Placement constraints that never change a job's computational identity."""

    max_cost_usd: float | None = field(default=None, metadata={"identity": False})
    deadline_utc: str | None = field(default=None, metadata={"identity": False})
    max_minutes: int | None = field(default=None, metadata={"identity": False})


__all__ = [
    "Admission",
    "CanonicalError",
    "ConfigError",
    "Gpu",
    "dumps",
    "from_mapping",
    "gpus",
    "load",
    "sha256",
    "value",
]
