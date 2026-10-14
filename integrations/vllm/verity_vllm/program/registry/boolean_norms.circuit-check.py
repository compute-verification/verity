"""circuit-check's bindings of `verity_vllm.program.registry.boolean_norms`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The served norms on bits at N = 16 with their MUFU statics at the word ids and at mufu's Boolean versions (the
    switch's binding, `verity_vllm.program.boolean`), the fused norm's lanes at two elements per thread, and the word
    Definitions of its constant steps."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import mufu as BM
    from verity_vllm.program.registry import boolean_norms as BN

    return [
        bind(BN.RMSNormFusedCuda, N=16, EPS=1e-5, **BN.CUDA_WORD),
        bind(BN.RMSNormTriton, N=16, EPS=1e-5, **BN.TRITON_WORD),
        bind(BN.RMSNormLanesCuda, N=40, BLOCK=32),
        bind(BN.RMSNormVarEpsCudaWord, N=16, EPS=1e-5),
        bind(BN.RMSNormVarEpsTritonWord, N=16, EPS=1e-5),
        bind(BN.RMSNormFusedCuda, N=16, EPS=1e-5, RSQRT=bind(BM.RsqrtApprox)),
        bind(BN.RMSNormTriton, N=16, EPS=1e-5, SQRT=bind(BM.MufuSqrtFtz), SCALEA=bind(BM.DivFullScaleA), RCP=bind(BM.DivFullRcp)),
        bind(BN.RMSNormCudaVector8, N=16, EPS=1e-5, BLOCK=2, **BN.CUDA_WORD),
        bind(BN.RMSNormFusedCudaVector8, N=16, EPS=1e-5, BLOCK=16, **BN.CUDA_WORD),
        bind(BN.RMSNormCudaVector8, N=16, EPS=1e-5, BLOCK=2, RSQRT=bind(BM.RsqrtApprox)),
        bind(BN.RMSNormFusedCudaVector8, N=16, EPS=1e-5, BLOCK=16, RSQRT=bind(BM.RsqrtApprox)),
        bind(BN.RMSNormCudaVector8, N=16, EPS=1e-6, BLOCK=2, **BN.CUDA_WORD),
        bind(BN.RMSNormFusedCudaVector8, N=16, EPS=1e-6, BLOCK=16, **BN.CUDA_WORD),
        bind(BN.RMSNormCudaVector8, N=16, EPS=1e-6, BLOCK=2, RSQRT=bind(BM.RsqrtApprox)),
        bind(BN.RMSNormFusedCudaVector8, N=16, EPS=1e-6, BLOCK=16, RSQRT=bind(BM.RsqrtApprox)),
        bind(BN.RMSNormVectorLanesCuda, N=24, BLOCK=2, FUSED=False),
        bind(BN.RMSNormVectorLanesCuda, N=24, BLOCK=2, FUSED=True),
    ]
