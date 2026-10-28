"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_softcap`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """FA2 with logit soft-capping on bits at one head of 17 keys, two blocks (a full one and a partial one) with Check_inf
    in both, its ``TANH`` and ``EX2`` at mufu's Boolean versions, which reaches its head and blocks; and the same with the two
    words inputs, its first block below MASKED_FROM."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import attention as BAT
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.circuits.ops.boolean import mufu as BM
    from verity.ml.definitions.registry import boolean_softcap as BSC

    return [
        bind(BSC.AttentionSoftcap, T=17, NH=1, KVH=1, D=16, BN=16, CAP=50.0, DOT=BG.HopperBF16WgmmaDot16, INV=BAT.Fa2InvSum,
             MASKED_FROM=0, TANH=BM.MufuTanh, EX2=BM.MufuEx2Ftz),
        bind(BSC.AttentionSoftcapIn, T=17, NH=1, KVH=1, D=16, BN=16, DOT=BG.HopperBF16WgmmaDot16, INV=BAT.Fa2InvSum,
             MASKED_FROM=1, TANH=BM.MufuTanh, EX2=BM.MufuEx2Ftz),
    ]
