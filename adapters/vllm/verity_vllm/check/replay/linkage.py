"""Each request's arrival step of record (`arrivals_of_record`)."""
from __future__ import annotations

from typing import Any

__all__ = []


def arrivals_of_record(declared: dict[str, int], schedule: dict[str, Any] | None) -> tuple[dict[str, int], bool]:
    """The arrivals every C2 consumer (population placement, sampled replay, boundary linkage) counts from: the ENGINE
    schedule of record's observed arrive steps when the population was placed on the observed
    committed-step axis (`schedule.use == "observed"`: every declared request observed, contiguous, exactly its max_tokens), else the declaration.
    Returns (arrivals, moved) -- `moved` iff an arrival differs from the declaration.  A missing / failed schedule leaves the declaration."""
    out = {str(k): int(v) for k, v in (declared or {}).items()}
    if not isinstance(schedule, dict) or schedule.get("use") != "observed":
        return out, False
    observed = schedule.get("observed_arrive_steps")                            # a schedule that states none moves no arrival
    obs = {str(k): int(v) for k, v in (observed if observed else {}).items() if str(k) in out}
    moved = any(out[k] != v for k, v in obs.items())
    out.update(obs)
    return out, moved
