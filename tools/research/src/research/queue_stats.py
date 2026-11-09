"""Time helpers `lean_times` and `vqueue_compare` share: a UTC ISO time read, hours between two times, and a nearest-rank p90."""
from __future__ import annotations

from datetime import datetime


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _hours(a: datetime | None, b: datetime | None) -> float | None:
    return (b - a).total_seconds() / 3600 if a and b else None


def p90(vals: list[float]) -> float:
    vals = sorted(vals)
    return vals[min(len(vals) - 1, round(0.9 * (len(vals) - 1)))]
