"""circuit-check's bindings of `verity.definitions.registry.tp_combine`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The combine units at WORLD 8: the fused MNNVL's coordinate, one- and two-shot; its chunks 2 and 3 of 16 of a 64-element
    token, one- and two-shot (GLM-5.3's chunks of 96 are over `circuit_check.targets.MAX_GATES`); row 2 of torch's multimem
    all-reduce of 4 rows of 24 (16 elements per owner) at a chain for owners 0-3 and a balanced tree for 4-7, which meets owners
    3 and 4, so both trees; row 1 of vLLM's custom 2-stage all-reduce of 4 rows of 16 (one pack per owner), which meets
    owners 2 and 3; and a row of 32 as 4 chunk rows of 8."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import tp_allreduce as TA
    from verity.definitions.registry import tp_combine as TC

    trees = (TA.chain(8),) * 4 + ((((0, 1), (2, 3)), ((4, 5), (6, 7))),) * 4
    return [
        ("AllReduceResidualMnnvlCoordinate_v1:1", lambda: bind(TC.AllReduceResidualMnnvlCoordinate, WORLD=8, SHOT=1)),
        ("AllReduceResidualMnnvlCoordinate_v1:2", lambda: bind(TC.AllReduceResidualMnnvlCoordinate, WORLD=8, SHOT=2)),
        ("AllReduceResidualMnnvlAt_v1:1", lambda: bind(TC.AllReduceResidualMnnvlAt, WORLD=8, N=64, C=16, AT=2, SHOT=1)),
        ("AllReduceResidualMnnvlAt_v1:2", lambda: bind(TC.AllReduceResidualMnnvlAt, WORLD=8, N=64, C=16, AT=3, SHOT=2)),
        ("AllReduceMultimemRow_v1",
         lambda: bind(TC.AllReduceMultimemRow, WORLD=8, N=24, ROWS=4, ROW=2, TREES=trees, FTZ=False)),
        ("AllReduceCustomRow_v1", lambda: bind(TC.AllReduceCustomRow, WORLD=8, N=16, ROWS=4, ROW=1, STAGE=2)),
        ("RowChunks_v1", lambda: bind(TC.RowChunks, N=32, C=8)),
    ]
