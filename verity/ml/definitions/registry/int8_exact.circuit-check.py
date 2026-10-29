"""circuit-check's bindings of `verity.ml.definitions.registry.int8_exact`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The linear at K = 64 over two output rows and its quantizer Call (its product is `ops.gemm`'s `ScaledMmS8_v1`, checked there);
    one KV head's 3-key window at d = 32; one row of 33 keys (P V' over two k steps, the second zero-padded); and the primitives no
    composite here reaches whole."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import int8_exact as X

    scale = 32 ** -0.5
    return [
        ("Int8Linear_v1", lambda: bind(X.Int8Linear, K=64, N=2)),
        ("Int8RowQuant_v1", lambda: bind(X.Int8RowQuant, K=64)),
        # the same quantizer with x as 32 rows of 2
        ("Int8RowQuant_v3/r32", lambda: bind(X.Int8RowQuantV3, K=64, R=32)),
        ("Int8KvQuant_v1", lambda: bind(X.Int8KvQuant, S=3, D=32)),
        ("Int8LutAttentionRow_v1/s33", lambda: bind(X.Int8LutAttentionRow, D=32, S=33, SCALE=scale)),
        # the same row reading the registered `exp2_lut127` (`Int8LutSoftmax_v2`, `Exp2Lut127_v2`, `TableRead8x2_v1`)
        ("Int8LutAttentionRow_v2/s33", lambda: bind(X.Int8LutAttentionRowV2, D=32, S=33, SCALE=scale)),
    ]
