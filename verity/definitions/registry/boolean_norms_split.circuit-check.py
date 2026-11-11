"""circuit-check's bindings of `verity.definitions.registry.boolean_norms_split`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The vector-8 paths' sums of squares at N = 16 with RSQRT at its word id and at mufu's Boolean version (as
    `boolean_norms`' whole norms), and their apply units at G = 8."""
    from verity.circuits.defs import bind
    from verity.definitions.ops.boolean import mufu as BM
    from verity.definitions.registry import boolean_norms as BN, boolean_norms_split as BS

    return [
        bind(BS.RMSNormSumSqCudaVector8, N=16, EPS=1e-5, BLOCK=2, **BN.CUDA_WORD),
        bind(BS.RMSNormFusedSumSqCudaVector8, N=16, EPS=1e-5, BLOCK=16, **BN.CUDA_WORD),
        bind(BS.RMSNormSumSqCudaVector8, N=16, EPS=1e-5, BLOCK=2, RSQRT=bind(BM.RsqrtApprox)),
        bind(BS.RMSNormFusedSumSqCudaVector8, N=16, EPS=1e-5, BLOCK=16, RSQRT=bind(BM.RsqrtApprox)),
        bind(BS.RMSNormApplyCudaVector8, G=8),
        bind(BS.RMSNormFusedApplyCudaVector8, G=8),
    ]
