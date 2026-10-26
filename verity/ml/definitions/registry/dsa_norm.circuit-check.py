"""circuit-check's bindings of `verity.ml.definitions.registry.dsa_norm`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`DsaRMSNorm_v1` at the two rows GLM-5.3 runs: q_a_layernorm (2048) and kv_a_layernorm (512), eps 1e-5; `TritonBlockRMSNorm_v1` at
    DeepSeek-V4's (q 1536 and kv 512 of a 2048 block on 8 warps, eps 1e-6)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import dsa_norm

    return [
        ("DsaRMSNorm_v1/q", lambda: bind(dsa_norm.DsaRMSNorm, N=2048, EPS=1e-5)),
        ("DsaRMSNorm_v1/kv", lambda: bind(dsa_norm.DsaRMSNorm, N=512, EPS=1e-5)),
        ("TritonBlockRMSNorm_v1/q", lambda: bind(dsa_norm.TritonBlockRMSNorm, N=1536, BLOCK=2048, WARPS=8, EPS=1e-6)),
        ("TritonBlockRMSNorm_v1/kv", lambda: bind(dsa_norm.TritonBlockRMSNorm, N=512, BLOCK=2048, WARPS=8, EPS=1e-6)),
    ]
