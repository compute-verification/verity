"""circuit-check's bindings of `verity.ml.definitions.registry.b1_norm_split`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _rmsnorm_fused_add_roots() -> list[Any]:
    """`RMSNormFusedCuda_v2` cut at its residual add: Llama-3.2-1B's norm after down_proj (its 8 add parts of 256 and the tail),
    and N = 64 in 4 parts."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import b1_norm_split as NS

    return [*(bind(NS.RMSNormFusedCudaAdd, N=N, C=C, AT=j) for N, C in ((2048, 256), (64, 16)) for j in range(N // C)),
            *(bind(NS.RMSNormFusedCudaTail, N=N, EPS=1e-5) for N in (2048, 64))]
