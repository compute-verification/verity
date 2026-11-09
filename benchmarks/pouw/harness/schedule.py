"""The harness's rules as pure functions, so they're tested without a GPU: how many operand sets a shape gets, a burst's
length, each rep's randomized order, the gate ledger that refuses to time what hasn't passed its gate in this run, and the
consistency floor."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field

MIN_SETS, MAX_SETS, MIN_BURST = 2, 512, 4
MIN_REPS = 20
#: the consistency gate's kernel-efficiency floor e: a hash-free call issuing Σ_p macs_p tensor MACs takes at least
#: e · Σ_p (macs_p / mkn) · (the fastest plain precision-p GEMM), or work was skipped (or the baseline isn't the fastest)
EFFICIENCY_FLOOR = 0.95
#: where the consistency gate applies: m·k·n at least this (smaller shapes are launch- or memory-bound)
COMPUTE_BOUND_MKN = 1 << 33


def set_count(weight_bytes: int, l2_bytes: int) -> int:
    """Operand sets for a shape whose weight (B, with its scales) takes `weight_bytes`: enough that the distinct weights pass
    twice the L2, at least 2 and at most 512."""
    if weight_bytes <= 0:
        raise ValueError("weight_bytes must be positive")
    return min(MAX_SETS, max(MIN_SETS, math.ceil(2 * l2_bytes / weight_bytes)))


def burst_len(sets: int) -> int:
    """Calls per burst: each operand set once, and at least MIN_BURST."""
    return max(sets, MIN_BURST)


def rep_orders(items: list, reps: int, seed: int) -> list[list]:
    """Each rep's order of `items`, a seeded shuffle per rep (the seed is recorded with the run)."""
    if reps < MIN_REPS:
        raise ValueError(f"at least {MIN_REPS} reps, not {reps}")
    rng = random.Random(seed)
    out = []
    for _ in range(reps):
        order = list(items)
        rng.shuffle(order)
        out.append(order)
    return out


class GateError(RuntimeError):
    pass


@dataclass
class GateLedger:
    """Every gate of the run, by key (an arm's (shape, flags role) or a baseline's (shape, family, kernel)). `require`
    refuses to time a key whose gate hasn't passed in this run; a failed gate is kept, and nothing is timed for its key."""

    records: dict = field(default_factory=dict)

    def record(self, key: tuple, ok: bool, detail: dict | None = None, wall_s: float | None = None) -> None:
        """`wall_s`: the host's seconds in this run of the gate, summed into the key's over every run of it."""
        if key in self.records and self.records[key]["ok"] and not ok:
            raise GateError(f"{key}: a gate failed after one passed")
        before = self.records.get(key, {}).get("wall_s")
        self.records[key] = {"ok": bool(ok), "detail": detail or {}}
        if wall_s is not None:
            self.records[key]["wall_s"] = (before or 0.0) + wall_s

    def passed(self, key: tuple) -> bool:
        return self.records.get(key, {}).get("ok", False)

    def require(self, key: tuple) -> None:
        if key not in self.records:
            raise GateError(f"{key}: not gated in this run; nothing is timed before its gate")
        if not self.records[key]["ok"]:
            raise GateError(f"{key}: its gate failed; nothing is timed")


def consistency(nohash_ms: float, tensor_macs: dict[str, int], mkn: int, best_ms: dict[str, float],
                efficiency: float = EFFICIENCY_FLOOR) -> dict | None:
    """The consistency gate at a compute-bound shape (None elsewhere, or when a precision the arm issues has no baseline):
    the hash-free call against e · Σ_p (macs_p / mkn) · best_ms[p]."""
    if mkn < COMPUTE_BOUND_MKN or any(p not in best_ms for p in tensor_macs):
        return None
    floor = efficiency * sum(macs / mkn * best_ms[p] for p, macs in tensor_macs.items())
    return {"nohash_ms": nohash_ms, "floor_ms": floor, "efficiency": efficiency, "ok": nohash_ms >= floor}


def summary(xs: list[float]) -> dict:
    """Median and spread of one entry's per-call times over the reps."""
    if not xs:
        raise ValueError("no reps")
    q = statistics.quantiles(xs, n=10, method="inclusive") if len(xs) > 1 else [xs[0]] * 9
    return {"median": statistics.median(xs), "p10": q[0], "p90": q[-1], "min": min(xs), "max": max(xs), "n": len(xs)}
