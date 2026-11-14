"""`verity_vllm.pipeline.telemetry.admission.predict`: how many forked replay workers a host limit admits when the
largest-VU workers do not all fit.

Without the freeze before the fork, the first two forked workers may each hold the largest VU, and the planner admits
them first: none while the spare room at the replay is below one of them, one from there until both fit. The limit at
which the first one fits is found by bisection on `host_limit_bytes`, and the size of one is read from the plan's own
term for two workers, so the test states no calibrated number."""
from __future__ import annotations

from typing import Any

from verity_vllm.pipeline.telemetry import admission as adm

MiB = adm.MiB


def row(**more: Any) -> adm.RowScale:
    return adm.RowScale(**{"model": "llama32-1b", "requests": 8, "prompt_tokens": 3819, "decode_rows": 494, "steps": 129,
                           "max_context": 1152, "pairs": 3, "fork_gc_freeze": False} | more)


def plan(limit: int | None, workers: int = 32) -> dict[str, Any]:
    return adm.predict(row(replay_workers=workers, host_limit_bytes=limit))["commit"]


def admitted(limit: int) -> int:
    return plan(limit)["replay_workers_admissible"]


def largest_worker() -> int:
    """One largest-VU worker's bytes: two workers are both largest-VU workers, so their term is two of them."""
    term = next(t for t in plan(None, workers=2)["terms"] if t["name"] == "commit.c2.replay_workers")
    assert term["bytes"] % 2 == 0
    return term["bytes"] // 2


def first_admitting() -> int:
    """The smallest host limit that admits a worker (a limit of 0 is none)."""
    lo, hi = 1, 1 << 42
    assert admitted(lo) == 0 and admitted(hi) >= 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if admitted(mid) == 0 else (lo, mid)
    return hi


def test_one_largest_worker_is_admitted_from_where_it_fits_until_both_do() -> None:
    first, big = first_admitting(), largest_worker()
    assert big > 250 * MiB
    assert [admitted(first - 1), admitted(first), admitted(first + big - 1), admitted(first + big)] == [0, 1, 1, 2]


def test_below_room_for_both_the_plan_names_the_largest_workers_size() -> None:
    got = plan(first_admitting())
    assert (got["replay_workers_admissible"], got["replay_workers_requested"]) == (1, 32)
    assert got["replay_workers_note"] == (f"two forked workers may hold the largest VU ({largest_worker() // MiB} MiB each); pass "
                                          "--replay-workers <= admissible or evaluate the largest strata in the parent")
