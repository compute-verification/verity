"""circuit-check bindings for the served non-batch-invariant CUDA RMSNorm word Definitions."""
from __future__ import annotations

from typing import Any


def _cuda_rmsnorm_roots() -> list[Any]:
    """Both kernels at every block size served for hidden widths 6144 and 7168."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import cuda_rmsnorm as C

    eps = 1e-5
    return [
        bind(C.RMSNormCudaVector8, N=6144, EPS=eps, BLOCK=768),
        bind(C.RMSNormCudaVector8, N=6144, EPS=eps, BLOCK=256),
        bind(C.RMSNormCudaVector8, N=7168, EPS=eps, BLOCK=896),
        bind(C.RMSNormCudaVector8, N=7168, EPS=eps, BLOCK=256),
        bind(C.RMSNormFusedCudaVector8, N=6144, EPS=eps, BLOCK=1024),
        bind(C.RMSNormFusedCudaVector8, N=6144, EPS=eps, BLOCK=256),
        bind(C.RMSNormFusedCudaVector8, N=7168, EPS=eps, BLOCK=1024),
        bind(C.RMSNormFusedCudaVector8, N=7168, EPS=eps, BLOCK=256),
    ]
