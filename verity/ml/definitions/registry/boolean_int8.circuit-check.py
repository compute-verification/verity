"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_int8`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """σ of each `int8_exact` composite with statics, at `int8_exact.circuit-check.py`'s statics or smaller: the linear at K = 32
    over two output rows, one KV head's 3-key window at d = 32, and one row of 33 keys (two k steps, the second zero-padded)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_int8 as X

    scale = 32 ** -0.5
    return [
        ("Int8RowScale_v2", lambda: bind(X.Int8RowScaleV2, K=4)),
        ("Int8StaticQuant_v2", lambda: bind(X.Int8StaticQuantV2, K=4)),
        ("Int8RowQuant_v2", lambda: bind(X.Int8RowQuantV2, K=4)),
        # the same quantizer with x as 2 rows of 2
        ("Int8RowQuant_v4/r2", lambda: bind(X.Int8RowQuantV4, K=4, R=2)),
        ("Int8Linear_v2", lambda: bind(X.Int8LinearV2, K=32, N=2)),
        ("Int8AttnScale_v2", lambda: bind(X.Int8AttnScaleV2, SCALE=scale)),
        ("Int8LutSoftmax_v3", lambda: bind(X.Int8LutSoftmaxV3, S=3)),
        ("Int8ColumnQuant_v2", lambda: bind(X.Int8ColumnQuantV2, S=3, D=2)),
        ("Int8KvQuant_v2", lambda: bind(X.Int8KvQuantV2, S=3, D=32)),
        ("Int8LutAttentionRow_v3/s33", lambda: bind(X.Int8LutAttentionRowV3, D=32, S=33, SCALE=scale)),
    ]
