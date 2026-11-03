"""circuit-check's bindings of `verity.ml.definitions.registry.linear_cut`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The A row at Llama-3.2-1B's o_proj (K = 2,048 read as its 32 heads' rows, its layer's call 1) and at one row of a small
    K; y at down_proj's row (N = 2,048 in rows of 64, call 3) and at gate_up's, the widest (N = 16,384 in rows of 256,
    call 2); each word Definition and its σ."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import linear_cut as LC

    return [
        ("LinearARow_v1", lambda: bind(LC.LinearARow, K=2048, R=32, ROUND=1, CALL=1)),
        ("LinearARow_v1/row", lambda: bind(LC.LinearARow, K=64, R=1, ROUND=0, CALL=0)),
        ("LinearOut_v1", lambda: bind(LC.LinearOut, N=2048, W=64, ROUND=0, CALL=3)),
        ("LinearOut_v1/wide", lambda: bind(LC.LinearOut, N=16384, W=256, ROUND=0, CALL=2)),
        ("LinearARow_v2", lambda: bind(LC.LinearARowV2, K=2048, R=32, ROUND=1, CALL=1)),
        ("LinearARow_v2/row", lambda: bind(LC.LinearARowV2, K=64, R=1, ROUND=0, CALL=0)),
        ("LinearOut_v2", lambda: bind(LC.LinearOutV2, N=2048, W=64, ROUND=0, CALL=3)),
        ("LinearOut_v2/wide", lambda: bind(LC.LinearOutV2, N=16384, W=256, ROUND=0, CALL=2)),
    ]
