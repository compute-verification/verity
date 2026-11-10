"""circuit-check's bindings of `verity.definitions.registry.boolean_dsa_norm`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`DsaRMSNorm_v2` at its smallest row (two elements a thread, one FMA each); `TritonBlockRMSNorm_v2` at a two-warp block whose
    second warp is half masked (48 of 64 threads live, two elements each); `DsaLatentRow_v2` over that smallest norm and GLM-5.3's
    k_pe head (64)."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_dsa_norm

    return [
        ("DsaRMSNorm_v2", lambda: bind(boolean_dsa_norm.DsaRMSNormV2, N=256, EPS=1e-5)),
        ("TritonBlockRMSNorm_v2", lambda: bind(boolean_dsa_norm.TritonBlockRMSNormV2, N=96, BLOCK=128, WARPS=2, EPS=1e-6)),
        ("DsaLatentRow_v2", lambda: bind(boolean_dsa_norm.DsaLatentRowV2, KVL=256, R=64, EPS=1e-5)),
    ]
