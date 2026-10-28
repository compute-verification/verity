"""circuit-check's bindings of `verity.ml.definitions.registry.value_inputs`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import b1, hopper
    from verity.ml.definitions.registry import value_inputs as VI

    return [
        # the Definitions whose program values are inputs (eps, the attention scale); the chain's second block rescales (T = BN + 1)
        ("RMSNormTriton_v3", lambda: bind(VI.RMSNormTritonV3, N=64)),
        ("RMSNormFusedCuda_v4", lambda: bind(VI.RMSNormFusedCudaV4, N=64)),
        ("Attention_v10", lambda: bind(VI.AttentionV10, T=17, NH=1, KVH=1, D=16, BN=16)),
        ("Attention_v12", lambda: bind(VI.AttentionV12, T=17, NH=1, KVH=1, D=16, BN=16, DOT=hopper.HopperBF16WgmmaDot16, INV=b1.P.Fa2InvSum)),
        ("Attention_v13", lambda: bind(VI.AttentionV13, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1, DOT=hopper.HopperBF16WgmmaDot16,
                                       INV=hopper.Fa3InvSum)),
        ("AttentionSoftcap_v4", lambda: bind(VI.AttentionSoftcapV4, T=17, NH=1, KVH=1, D=16, BN=16)),
        ("AttentionSoftcap_v5", lambda: bind(VI.AttentionSoftcapV5, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1,
                                             DOT=hopper.HopperBF16WgmmaDot16, INV=b1.P.Fa2InvSum)),
    ]
