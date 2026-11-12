"""circuit-check's bindings of `verity.definitions.registry.cuda_rmsnorm_split`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _cuda_rmsnorm_split_roots() -> list[Any]:
    """Each kernel's sum of squares at every block size served for hidden widths 6144 and 7168, and the apply units at
    G = 64 and 96."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import cuda_rmsnorm_split as NS

    return [
        *(bind(NS.RMSNormSumSqCudaVector8, N=N, EPS=eps, BLOCK=block)
          for N, eps, blocks in ((6144, 1e-5, (768, 256)), (7168, 1e-6, (896, 256))) for block in blocks),
        *(bind(NS.RMSNormFusedSumSqCudaVector8, N=N, EPS=eps, BLOCK=block)
          for N, eps in ((6144, 1e-5), (7168, 1e-6)) for block in (1024, 256)),
        *(bind(fn, G=G) for fn in (NS.RMSNormApplyCudaVector8, NS.RMSNormFusedApplyCudaVector8) for G in (64, 96)),
    ]
