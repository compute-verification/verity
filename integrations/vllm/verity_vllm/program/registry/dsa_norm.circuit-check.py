"""circuit-check's bindings of `verity_vllm.program.registry.dsa_norm`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`DsaRMSNorm_v1` at the two rows GLM-5.3 runs: q_a_layernorm (2048) and kv_a_layernorm (512), eps 1e-5."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dsa_norm

    return [
        ("DsaRMSNorm_v1/q", lambda: bind(dsa_norm.DsaRMSNorm, N=2048, EPS=1e-5)),
        ("DsaRMSNorm_v1/kv", lambda: bind(dsa_norm.DsaRMSNorm, N=512, EPS=1e-5)),
    ]
