"""circuit-check's bindings of `verity_vllm.program.registry.b1`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from circuit_check.targets import TINY
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import b1

    return [
        ("Serve_v1", lambda: bind(b1.Serve, C=TINY, LP=3, STEPS=2)),
        ("Serve_v2", lambda: b1.serve_v2(TINY, 3, 2)),
        ("Serve_v4", lambda: bind(b1.ServeV4, C=TINY, LP=3, STEPS=2, BN=16)),
        ("RMSNormFusedCuda_v1", lambda: bind(b1.RMSNormFusedCudaV1, N=64, EPS=1e-5)),
        ("ResidualAdd_v1", lambda: bind(b1.ResidualAdd, N=16)),
    ]
