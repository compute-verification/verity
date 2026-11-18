"""Who ran a check a benchmark prints or stores: its ``checked_by``, one of proofs' runners (`RAN_BY` in
`benchmarks/one_stage/served_verdicts.py`), never a word of our own. Every check our benchmarks run is on the developer's
side, on the node that served or proved: none is the auditor's. A stored result that predates the field says
`NOT_RECORDED`, never a runner read into it.

- `CHECKED_BY`: each runner, in the words of where our benchmarks run it.
- `words(value)`: the printed ``checked by: RUNNER (WORDS)``, or ``checked by: not recorded``.
- `of(checks)`: the one runner a set of stored checks (gates, verdicts) names, None where any lacks it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

#: proofs' runners (`served_verdicts.RAN_BY`'s keys), each in the words of where our benchmarks run it
CHECKED_BY = {"piece": "run by the developer's side on the served node: the benchmark's own process or one it starts "
                       "there; no auditor-side entry runs it",
              "lean": "recomputed by the Lean verifier in the run's own job, on the developer's node",
              "device": "run by an auditor's device"}
PIECE, LEAN = "piece", "lean"
#: what a result that predates the field says, and the table cell of one
NOT_RECORDED = "not recorded"


def words(value: str | None) -> str:
    if value is None:
        return f"checked by: {NOT_RECORDED}"
    if value not in CHECKED_BY:
        raise ValueError(f"checked_by {value!r}: one of {', '.join(map(repr, CHECKED_BY))}")
    return f"checked by: {value} ({CHECKED_BY[value]})"


def of(checks: Iterable[Mapping[str, Any]]) -> str | None:
    said = {c.get("checked_by") for c in checks}
    return said.pop() if len(said) == 1 else None
