"""circuit-check's bindings of `verity.ml.definitions.registry.tp_combine`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The combine units at WORLD 8: the fused MNNVL's coordinate, one- and two-shot; row 2 of torch's multimem all-reduce of 4
    rows of 24 (16 elements per owner) at a chain for owners 0-3 and a balanced tree for 4-7, which meets owners 3 and 4, so
    both trees; and row 1 of vLLM's custom 2-stage all-reduce of 4 rows of 16 (one pack per owner), which meets owners 2 and
    3."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import tp_allreduce as TA
    from verity.ml.definitions.registry import tp_combine as TC

    trees = (TA.chain(8),) * 4 + ((((0, 1), (2, 3)), ((4, 5), (6, 7))),) * 4
    return [
        ("AllReduceResidualMnnvlCoordinate_v1:1", lambda: bind(TC.AllReduceResidualMnnvlCoordinate, WORLD=8, SHOT=1)),
        ("AllReduceResidualMnnvlCoordinate_v1:2", lambda: bind(TC.AllReduceResidualMnnvlCoordinate, WORLD=8, SHOT=2)),
        ("AllReduceMultimemRow_v1",
         lambda: bind(TC.AllReduceMultimemRow, WORLD=8, N=24, ROWS=4, ROW=2, TREES=trees, FTZ=False)),
        ("AllReduceCustomRow_v1", lambda: bind(TC.AllReduceCustomRow, WORLD=8, N=16, ROWS=4, ROW=1, STAGE=2)),
    ]
