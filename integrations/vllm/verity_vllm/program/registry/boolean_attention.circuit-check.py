"""circuit-check's bindings of `verity_vllm.program.registry.boolean_attention`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_attention

    return [
        # about 7 minutes a target: the smallest Boolean chain whose second block rescales (T = BN + 1)
        ("Attention_v6", lambda: bind(boolean_attention.AttentionV6, T=17, NH=1, KVH=1, D=16, BN=16)),
        ("Attention_v7", lambda: bind(boolean_attention.AttentionV7, T=5, NH=1, KVH=1, D=16, BN=16, **boolean_attention.ATTENTION_FA3)),
        # the served FA2 binding at the smallest chain whose unguarded block rescales (T = BN + 1, block 0 below MASKED_FROM)
        ("Attention_v8", lambda: bind(boolean_attention.AttentionV8, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1,
                                      DOT=boolean_attention.ATTENTION_FA3["DOT"], INV=boolean_attention.ATTENTION_FA2["INV"])),
        ("Attention_v9", lambda: bind(boolean_attention.AttentionV9, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1,
                                      **boolean_attention.ATTENTION_FA3)),
    ]
