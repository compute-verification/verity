"""circuit-check's bindings of `verity_vllm.program.registry.boolean_dsa_norm`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`DsaRMSNorm_v2` at its smallest row (two elements a thread, one FMA each)."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_dsa_norm

    return [
        ("DsaRMSNorm_v2", lambda: bind(boolean_dsa_norm.DsaRMSNormV2, N=256, EPS=1e-5)),
    ]
