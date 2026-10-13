"""circuit-check's bindings of `verity_vllm.program.registry.boolean_mhc`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_mhc as BM

    e = dict(EPS=1e-6, HC_EPS=1e-6)
    # two streams, the smallest blocks each kernel's check accepts, two Sinkhorn iterations and two splits; each big fuse's Calls
    # are the mixes (K 32 in both) and the norm bound beside it, the TF32 GEMM's its split, so each flat circuit is traced once
    return [
        ("HopperTf32WgmmaDot8_v2", lambda: bind(BM.HopperTf32WgmmaDot8V2)),
        ("MhcGpuF32Add_v1", lambda: bind(BM.MhcGpuF32Add)),
        ("MhcGpuF32Mul_v1", lambda: bind(BM.MhcGpuF32Mul)),
        ("MhcGpuF32Fma_v1", lambda: bind(BM.MhcGpuF32Fma)),
        ("MhcSigmoid_v2", lambda: bind(BM.MhcSigmoidV2)),
        ("MhcSinkhorn_v2", lambda: bind(BM.MhcSinkhornV2, HC=2, REPEAT=2, EPS=1e-6)),
        ("MhcMixes_v2", lambda: bind(BM.MhcMixesV2, HC=2, NS=2, K=32, REPEAT=2, **e)),
        ("MhcPreNorm_v2", lambda: bind(BM.MhcPreNormV2, HC=2, H=16, TN=1, EPS=1e-6)),
        ("MhcPreNormBroadcast_v2", lambda: bind(BM.MhcPreNormBroadcastV2, HC=2, H=32, TN=2, EPS=1e-6)),
        ("MhcPreBigFuse_v2", lambda: bind(BM.MhcPreBigFuseV2, HC=2, H=16, NS=2, TN=1, REPEAT=2, **e)),
        ("MhcPreBigFuseBroadcast_v2", lambda: bind(BM.MhcPreBigFuseBroadcastV2, HC=2, H=32, NS=2, TN=2, REPEAT=2, **e)),
        ("MhcPost_v2", lambda: bind(BM.MhcPostV2, HC=2, N=2)),
        ("MhcFusedSplit_v2", lambda: bind(BM.MhcFusedSplitV2, HC=2, N=2, OUT=8, TH=2)),
        ("MhcPrenormTf32Split_v2", lambda: bind(BM.MhcPrenormTf32SplitV2, K=64, OUT=1)),
        ("MhcPrenormTf32_v2", lambda: bind(BM.MhcPrenormTf32V2, K=128, NS=2, OUT=1)),
        ("HcHead_v2", lambda: bind(BM.HcHeadV2, HC=2, H=8, TH=1, **e)),
    ]
