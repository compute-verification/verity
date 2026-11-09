"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_fa3_split`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_attention, boolean_fa3_split

    return [
        # the smallest split chain: two splits of one block each (T = BN + 1), the full block below MASKED_FROM
        ("AttentionSplit_v2", lambda: bind(boolean_fa3_split.AttentionSplitV2, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1, PER=1,
                                           **boolean_attention.ATTENTION_FA3)),
    ]
