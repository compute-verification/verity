"""circuit-check's bindings of `verity.definitions.registry.boolean_tp_combine`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The combine units on bits at WORLD 8: the fused MNNVL's coordinate, one- and two-shot; its chunks 2 and 3 of 8 of a
    32-element token, one- and two-shot (at 16 elements a chunk both are over `MAX_GATES`); row 2 of torch's multimem all-reduce
    of 4 rows of 24 (16 elements per owner) at a chain for owners 0-3 and a balanced tree for 4-7, which meets both; row 1 of
    vLLM's custom 2-stage all-reduce of 4 rows of 16 (one pack per owner), which meets owners 2 and 3; and a row of 32 as 4
    chunk rows of 8 (no gate).  The multimem row is over
    `circuit_check.targets.MAX_GATES` at either FTZ, so `select` would keep only the smaller, IEEE one: FTZ changes only the
    vectors it calls, whose FTZ form `boolean_tp_allreduce`'s sidecar pins."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_tp_combine as BC
    from verity.definitions.registry import tp_allreduce as TA

    trees = (TA.chain(8),) * 4 + ((((0, 1), (2, 3)), ((4, 5), (6, 7))),) * 4
    return [
        *[bind(BC.AllReduceResidualMnnvlCoordinate, WORLD=8, SHOT=s) for s in (1, 2)],
        *[bind(BC.AllReduceResidualMnnvlAt, WORLD=8, N=32, C=8, AT=1 + s, SHOT=s) for s in (1, 2)],
        bind(BC.AllReduceMultimemRow, WORLD=8, N=24, ROWS=4, ROW=2, TREES=trees, FTZ=False),
        bind(BC.AllReduceCustomRow, WORLD=8, N=16, ROWS=4, ROW=1, STAGE=2),
        bind(BC.RowChunks, N=32, C=8),
    ]
