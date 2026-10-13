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
    return [
        # two streams, two splits, one warp, two Sinkhorn iterations (one each for the whole layers): the shapes of the word
        # bindings at the smallest statics their checks accept
        ("MhcSinkhorn_v2", lambda: bind(BM.MhcSinkhornV2, HC=2, REPEAT=2, EPS=1e-6)),
        ("MhcMixes_v2", lambda: bind(BM.MhcMixesV2, HC=2, NS=2, K=64, REPEAT=2, **e)),
        ("MhcPreNorm_v2", lambda: bind(BM.MhcPreNormV2, HC=2, H=32, TN=2, EPS=1e-6)),
        ("MhcPreNormBroadcast_v2", lambda: bind(BM.MhcPreNormBroadcastV2, HC=2, H=32, TN=2, EPS=1e-6)),
        ("MhcPreBigFuse_v2", lambda: bind(BM.MhcPreBigFuseV2, HC=2, H=32, NS=2, TN=2, REPEAT=2, **e)),
        ("MhcPreBigFuseBroadcast_v2", lambda: bind(BM.MhcPreBigFuseBroadcastV2, HC=2, H=32, NS=2, TN=2, REPEAT=2, **e)),
        ("MhcPost_v2", lambda: bind(BM.MhcPostV2, HC=2, N=4)),
        ("MhcFusedSplit_v2", lambda: bind(BM.MhcFusedSplitV2, HC=2, N=32, OUT=2, TH=32)),
        ("MhcFusedPostPre_v2", lambda: bind(BM.MhcFusedPostPreV2, HC=2, H=64, NS=2, TH=32, TN=2, REPEAT=1, **e)),
        ("MhcPrenormTf32Split_v2", lambda: bind(BM.MhcPrenormTf32SplitV2, K=64, OUT=2)),
        ("MhcPrenormTf32_v2", lambda: bind(BM.MhcPrenormTf32V2, K=128, NS=2, OUT=2)),
        ("MhcPostPreTf32_v2", lambda: bind(BM.MhcPostPreTf32V2, HC=2, H=32, NS=1, TN=2, REPEAT=1, **e)),
        ("MhcPreBroadcast_v2", lambda: bind(BM.MhcPreBroadcastV2, HC=2, H=64, NS=1, TN=2, REPEAT=1, **e)),
        ("HcHead_v2", lambda: bind(BM.HcHeadV2, HC=2, H=16, TH=2, **e)),
    ]
