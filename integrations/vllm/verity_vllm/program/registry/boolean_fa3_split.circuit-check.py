"""circuit-check's bindings of `verity_vllm.program.registry.boolean_fa3_split`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_attention as BA
    from verity_vllm.program.registry import boolean_fa3_split as BF

    fa3 = BA.ATTENTION_FA3
    return [
        ("AttnBlockFirst_v2", lambda: bind(BF.AttnBlockFirstV2, D=16, NVIS=8, BN=16, DOT=fa3["DOT"])),
        # T = 33 at BN = 16 in two splits: blocks 1 and 0 (block 0 below MASKED_FROM, unguarded), then block 2 of one key
        ("AttentionHeadSplit_v2", lambda: bind(BF.AttentionHeadSplitV2, T=33, NBM=3, NS=2, D=16, BN=16, MASKED_FROM=1, **fa3)),
        ("AttentionSplit_v2", lambda: bind(BF.AttentionSplitV2, T=33, NBM=3, NS=2, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1, **fa3)),
    ]
