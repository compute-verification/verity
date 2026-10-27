"""circuit-check's bindings of `verity.ml.definitions.registry.cublas_gemv`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The word Definitions: the dot at K 2048 over 16 lanes and at K 4096 over 32 (the served kernel's two block shapes), and
    the rows at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import cublas_gemv as GV

    return [
        ("CublasGemvDot_v1", lambda: bind(GV.CublasGemvDotWord, K=2048, LANES=16)),
        ("CublasGemvDot_v1:LANES=32", lambda: bind(GV.CublasGemvDotWord, K=4096, LANES=32)),
        ("CublasGemv_v1", lambda: bind(GV.CublasGemvWord, K=64, N=2, LANES=16)),
    ]


def _boolean_roots() -> list[Any]:
    """On bits: the dot over 16 lanes of four steps and over 32 of two, and the rows at two W rows."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import cublas_gemv as GV

    return [
        bind(GV.CublasGemvDot, K=64, LANES=16),
        bind(GV.CublasGemvDot, K=64, LANES=32),
        bind(GV.CublasGemv, K=64, N=2, LANES=16),
    ]
