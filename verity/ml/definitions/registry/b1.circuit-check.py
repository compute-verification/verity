"""circuit-check's bindings of `verity.ml.definitions.registry.b1`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from circuit_check.targets import TINY
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import b1

    return [
        ("Serve_v1", lambda: bind(b1.Serve, C=TINY, LP=3, STEPS=2)),
        ("Serve_v2", lambda: b1.serve_v2(TINY, 3, 2)),
        ("Serve_v4", lambda: bind(b1.ServeV4, C=TINY, LP=3, STEPS=2, BN=16)),
        ("RMSNormFusedCuda_v1", lambda: bind(b1.RMSNormFusedCudaV1, N=64, EPS=1e-5)),
        ("ResidualAdd_v1", lambda: bind(b1.ResidualAdd, N=16)),
        # greedy token selection cut along V: a chunk's step, a chunk, and a row of 200 in chunks of 64 (the last one 8 wide)
        ("TokenSelectChunkStep_v1", lambda: bind(b1.TokenSelectChunkStep)),
        ("TokenSelectChunk_v1", lambda: bind(b1.TokenSelectChunk, C=64)),
        ("TokenSelectCombine_v1", lambda: bind(b1.TokenSelectCombine, V=200, C=64)),
    ]
