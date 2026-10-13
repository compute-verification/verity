"""circuit-check's bindings of `verity_vllm.program.registry.mhc`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import mhc

    e = dict(EPS=1e-6, HC_EPS=1e-6)
    return [
        # DeepSeek-V4-Pro's served statics (HC 4, H 7168, Sinkhorn 20, the eps 1e-6): the Sinkhorn, the mixes at 4 splits (at most
        # 16 tokens), 112 (the TF32 GEMM at most 64 tokens) and 28 (layer 0's broadcast), the fused RMSNorm, the post, one split of
        # mhc_fused, one split and the whole of the TF32 prenorm GEMM, the broadcast pre and hc_head
        ("MhcSinkhorn_v1", lambda: bind(mhc.MhcSinkhorn, HC=4, REPEAT=20, EPS=1e-6)),
        ("MhcMixes_v1", lambda: bind(mhc.MhcMixes, HC=4, NS=4, K=28672, REPEAT=20, **e)),
        ("MhcMixes_v1/ns112", lambda: bind(mhc.MhcMixes, HC=4, NS=112, K=28672, REPEAT=20, **e)),
        ("MhcMixes_v1/broadcast", lambda: bind(mhc.MhcMixes, HC=4, NS=28, K=7168, REPEAT=20, **e)),
        ("MhcPreNorm_v1", lambda: bind(mhc.MhcPreNorm, HC=4, H=7168, TN=64, EPS=1e-6)),
        ("MhcPreNormBroadcast_v1", lambda: bind(mhc.MhcPreNormBroadcast, HC=4, H=7168, TN=64, EPS=1e-6)),
        ("MhcPreBigFuse_v1", lambda: bind(mhc.MhcPreBigFuse, HC=4, H=7168, NS=4, TN=64, REPEAT=20, **e)),
        ("MhcPreBigFuseBroadcast_v1", lambda: bind(mhc.MhcPreBigFuseBroadcast, HC=4, H=7168, NS=28, TN=64, REPEAT=20, **e)),
        ("MhcPost_v1", lambda: bind(mhc.MhcPost, HC=4, N=7168)),
        ("MhcFusedSplit_v1", lambda: bind(mhc.MhcFusedSplit, HC=4, N=1792, OUT=24, TH=256)),
        ("MhcPrenormTf32Split_v1", lambda: bind(mhc.MhcPrenormTf32Split, K=256, OUT=24)),
        ("MhcPrenormTf32_v1", lambda: bind(mhc.MhcPrenormTf32, K=7168, NS=28, OUT=24)),
        ("MhcPreBroadcast_v1", lambda: bind(mhc.MhcPreBroadcast, HC=4, H=7168, NS=28, TN=64, REPEAT=20, **e)),
        ("HcHead_v1", lambda: bind(mhc.HcHead, HC=4, H=7168, TH=128, **e)),
        # the two whole post + pre layers at hidden 1024 (4 splits of 256 elements, one a thread; the TF32 GEMM's 64 blocks in 4
        # splits): every callee is a served one's shape at a smaller H
        ("MhcFusedPostPre_v1", lambda: bind(mhc.MhcFusedPostPre, HC=4, H=1024, NS=4, TH=256, TN=64, REPEAT=20, **e)),
        ("MhcPostPreTf32_v1", lambda: bind(mhc.MhcPostPreTf32, HC=4, H=1024, NS=4, TN=64, REPEAT=20, **e)),
    ]
