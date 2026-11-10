"""The step schedule, as the verifier's `Spot.Schedule` defines it, which the difftest's ``second_schedule`` holds this
to: the replay blocks of a two-stage window as runs of positions of its requests, which the boundary registration's
record commits before the first coin (``window.schedule``), each run ``{"request", "first_span", "spans",
"first_position", "length"}``. Round 2 refuses a record whose schedule is no schedule of its replay blocks
(``second-schedule``, `second_stage`); what one is, and each block's positions (`at`), are the Lean's (its module
docstring)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

KEYS = ["first_position", "first_span", "length", "request", "spans"]


@dataclass(frozen=True)
class Run:
    """`Schedule.Run`: ``spans`` replay blocks from ``first_span``, each ``length`` positions of ``request``, the first
    from ``first_position``."""
    request: str
    first_span: int
    spans: int
    first_position: int
    length: int


def _nat(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _get(j: Any, k: str) -> Any:
    return j.get(k) if isinstance(j, dict) else None


def run_of(j: Any) -> Run | None:
    """`Schedule.Run.ofJson?`: exactly `KEYS`, the request a string and the rest naturals."""
    if not (isinstance(j, dict) and sorted(j) == KEYS):
        return None
    if not (isinstance(j["request"], str) and all(_nat(j[k]) for k in KEYS if k != "request")):
        return None
    return Run(j["request"], j["first_span"], j["spans"], j["first_position"], j["length"])


def runs_of(j: Any) -> list[Run] | None:
    """`Schedule.ofJson?`: a list of runs."""
    if not isinstance(j, list):
        return None
    runs = [run_of(r) for r in j]
    return None if any(r is None for r in runs) else runs


def why(population: int, runs: list[Run]) -> str | None:
    """`Schedule.why`: why ``runs`` are no schedule of ``population`` replay blocks; None if they are one."""
    block, nxt = 0, {}
    for r in runs:
        if r.spans == 0 or r.length == 0:
            return f"a run from block {r.first_span} of no block or no position"
        if r.first_span != block:
            return f"a run from block {r.first_span}, not from block {block}"
        if r.first_position != nxt.get(r.request, 0):
            return f"a run of {r.request} from position {r.first_position}, not from position {nxt.get(r.request, 0)}"
        block, nxt[r.request] = block + r.spans, r.first_position + r.spans * r.length
    return None if block == population else f"runs of {block} blocks, not the population's {population}"


def at(runs: list[Run], s: int) -> tuple[str, int, int] | None:
    """`Schedule.at?`: replay block ``s``'s request and its positions ``[a, b)``; None for a block no run holds."""
    for r in runs:
        if r.first_span <= s < r.first_span + r.spans:
            a = r.first_position + (s - r.first_span) * r.length
            return r.request, a, a + r.length
    return None


def failure(rec: Any) -> str | None:
    """`Schedule.failure`: why record ``rec``'s schedule (``window.schedule``) is refused; None for a record that commits
    none."""
    j = _get(_get(rec, "window"), "schedule")
    if j is None:
        return None
    runs, m = runs_of(j), _get(rec, "population")
    if runs is None:
        return "the schedule is not a list of runs"
    if not _nat(m):
        return "the record's population is not a natural number"
    return why(m, runs)
