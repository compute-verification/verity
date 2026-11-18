"""Times and quantiles the timing reports share (`run_times`, `lean_times`)."""
from __future__ import annotations

from datetime import UTC, datetime


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def rid_time(rid: str) -> datetime:
    return datetime.strptime(rid[1:16], "%Y%m%d-%H%M%S").replace(tzinfo=UTC)


def _hours(a: datetime | None, b: datetime | None) -> float | None:
    return (b - a).total_seconds() / 3600 if a and b else None


def pct(vals: list[float], q: float) -> float:
    """The nearest-rank ``q`` quantile of ``vals`` (not empty)."""
    vals = sorted(vals)
    return vals[min(len(vals) - 1, round(q * (len(vals) - 1)))]
