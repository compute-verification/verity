"""The rate arithmetic a verifier expands a rate sampling design with: the sample count at a rate, each stratum's floor, and
the work table, read and raised to the rate over the partition query's strata."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from verity.circuits import partition_query as P
from verity.circuits.program import Program


def rate_count(num: int, den: int, tables: int, n: int) -> int:
    """The rate sampling design's sample over n proof blocks: ceil(NUM n / DEN) rounded up to a multiple of `tables`, so each of the session's
    tables proves the same count and every proof block is sampled with probability K/n >= NUM/DEN, whatever n the prover registers.
    The Lean's `Spot.Rate.count` is this."""
    return tables * -(-(num * n) // (den * tables))


def rate_floor(num: int, den: int, n: int, floor: int) -> int:
    """A stratum's floor under the work rate NUM/DEN: ceil(NUM n / DEN), raised to the verifier's `floor`, at most the stratum's n proof
    blocks. A work sampling design samples each stratum at its floor or more (`work_k`), so every proof block of a stratum whose
    floor this is is sampled with probability at least NUM/DEN, whatever strata the prover registers. The Lean's
    `Spot.Rate.floor` is this, and `Rate.workK_ge` the bound."""
    return min(n, max(floor, rate_count(num, den, 1, n)))


def rate_table(program: Program, query: Mapping[str, Any], num: int, den: int,
               table: Mapping[str, Any] | None) -> dict[str, dict[str, int]]:
    """The work table the work rate sampling design `work-rate:NUM/DEN` is expanded over: each stratum of the program under the
    query (`partition.strata`) with the verifier's `table`'s work and its floor at the rate (`rate_floor`). Raises
    `partition.QueryRefused` when the query doesn't apply, and `ValueError` when there is no table or the table names no work
    for a stratum's template. The Lean's `Select.Rule.tableOver` is this."""
    if table is None:
        raise ValueError("a work rate law needs the verifier's work table")
    works = work_table(table)
    out: dict[str, dict[str, int]] = {}
    for t, rs in P.strata(program, query):
        if t not in works:
            raise ValueError(f"the work table names no work for template {t!r}")
        w, f = works[t]
        out[t] = {"work": w, "floor": rate_floor(num, den, _size(rs), f)}
    return out


def work_table(table: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """The verifier's work table (Lean `workTableOfJson`): template descriptor id -> the credited work of one of its proof blocks
    (a natural number, floor one) or `{"work": w, "floor": f}` with f >= 1; as `{template: (work, floor)}`. Raises
    `ValueError` naming the first bad entry. See the module docstring for what a proof block's work may count."""
    if not isinstance(table, Mapping):
        raise ValueError("a work table is an object from template id to work per unit")
    out: dict[str, tuple[int, int]] = {}
    for t, v in table.items():
        if not isinstance(t, str):
            raise ValueError("a work table is an object from template id to work per unit")
        if _nat(v):
            out[t] = (v, 1)
        elif isinstance(v, Mapping):
            if set(v) - {"work", "floor"} or not _nat(v.get("work")):
                raise ValueError(f"work table: template {t!r}: neither a work nor an object of work and floor")
            f = v.get("floor", 1)
            if not _nat(f):
                raise ValueError(f"work table: template {t!r}: neither a work nor an object of work and floor")
            if f < 1:
                raise ValueError(f"work table: template {t!r}: a floor is at least one")
            out[t] = (v["work"], f)
        else:
            raise ValueError(f"work table: template {t!r}: the work is not a natural number")
    return out


def _size(ranges: Sequence[Sequence[int]]) -> int:
    return sum(hi - lo for lo, hi in ranges)


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _nat(x: Any) -> bool:
    return _int(x) and x >= 0
