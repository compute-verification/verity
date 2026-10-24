"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_fa2_split`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at small statics: three splits combined; a head of T = 17 at BN = 16 in three splits (a full
    block, the partial one, an empty split); two heads of one KV head in two splits."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_fa2_split as BX

    fa2 = BX.ATTENTION_FA2_SM120
    return [
        ("Fa2SplitEnd_v2", lambda: bind(BX.Fa2SplitEndV2, D=16, INV=fa2["INV"], SCALE=0.25)),
        ("Fa2Combine_v2", lambda: bind(BX.Fa2CombineV2, NS=3, D=16)),
        ("Fa2AttentionHeadSplit_v2", lambda: bind(BX.Fa2AttentionHeadSplitV2, T=17, NS=3, D=16, BN=16, MASKED_FROM=1, **fa2)),
        ("Fa2AttentionSplit_v2", lambda: bind(BX.Fa2AttentionSplitV2, T=17, NS=2, NH=2, KVH=1, D=16, BN=16, MASKED_FROM=1, **fa2)),
    ]
