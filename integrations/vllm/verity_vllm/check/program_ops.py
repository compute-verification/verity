"""The operations X-09 (`batch_decomp.decompose`) and GM-01 (`global_match.check`) run on Programs, as one value passed explicitly."""
from __future__ import annotations

import dataclasses
from collections.abc import Callable


@dataclasses.dataclass(frozen=True)
class Ops:
    """The operations X-09 and GM-01 run on Programs.  `batch_decomp.record_ops()` = the checker of record's functions;
    `global_match_fast.ops()` = compute-once equals of them (MATCH_IMPL=fast): each returns what the record's function returns for the
    same arguments."""
    name: str
    load_oracle: Callable        # (fold dir, label) -> RootView
    load_derived: Callable       # (derived request-wrapper dir, label) -> RootView
    projection: Callable         # Projection's constructor signature -> Projection
    compare_steps: Callable      # compare.compare_steps(A, B, unfold_a)
    dag_hashes: Callable         # dag_hashes(P, only): instance -> equality key of its Merkle class
    live_cone: Callable          # live_cone(P)
    # decompose's per-request legs: None = in this process, in order (the record); else fn(items, leg, weight) ->
    # [leg(item) ...] in item order (the fast path: global_match_fast._x09_legs, forked workers)
    legs: Callable | None = None
