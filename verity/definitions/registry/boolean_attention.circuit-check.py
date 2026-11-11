"""circuit-check's bindings of `verity.definitions.registry.boolean_attention`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _cut_served_roots() -> list[Any]:
    """The units a served head is cut into, on bits, at D = 64 and BN = 192 on Hopper: `AttnBlockFirst_v3` at a first block of
    1, 35 and all 192 keys, and `AttnHeadEnd_v2`.  Each is over `targets.MAX_GATES`, so it holds a ``digests`` pin, which keeps
    it in ``--all``: the 192-key block is the largest a head has, and the only one here whose every k-step is whole."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_attention

    fa3 = boolean_attention.ATTENTION_FA3
    return [*(bind(boolean_attention.AttnBlockFirstV3, D=64, NVIS=nvis, BN=192, DOT=fa3["DOT"]) for nvis in (1, 35, 192)),
            bind(boolean_attention.AttnHeadEndV2, D=64, INV=fa3["INV"])]


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_attention

    return [
        # about 7 minutes a target: the smallest Boolean chain whose second block rescales (T = BN + 1)
        ("Attention_v6", lambda: bind(boolean_attention.AttentionV6, T=17, NH=1, KVH=1, D=16, BN=16)),
        ("Attention_v7", lambda: bind(boolean_attention.AttentionV7, T=5, NH=1, KVH=1, D=16, BN=16, **boolean_attention.ATTENTION_FA3)),
        # the served FA2 binding at the smallest chain whose unguarded block rescales (T = BN + 1, block 0 below MASKED_FROM)
        ("Attention_v8", lambda: bind(boolean_attention.AttentionV8, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1,
                                      DOT=boolean_attention.ATTENTION_FA3["DOT"], INV=boolean_attention.ATTENTION_FA2["INV"])),
        ("Attention_v9", lambda: bind(boolean_attention.AttentionV9, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1,
                                      **boolean_attention.ATTENTION_FA3)),
        # v6 to v9's configurations with the scale an input: the rescale reads the scale word (T = BN + 1)
        ("Attention_v14", lambda: bind(boolean_attention.AttentionV14, T=17, NH=1, KVH=1, D=16, BN=16)),
        ("Attention_v15", lambda: bind(boolean_attention.AttentionV15, T=5, NH=1, KVH=1, D=16, BN=16,
                                       **boolean_attention.ATTENTION_FA3)),
        ("Attention_v16", lambda: bind(boolean_attention.AttentionV16, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1,
                                       **boolean_attention.ATTENTION_FA3)),
        ("Attention_v17", lambda: bind(boolean_attention.AttentionV17, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1,
                                       DOT=boolean_attention.ATTENTION_FA3["DOT"], INV=boolean_attention.ATTENTION_FA2["INV"])),
    ]
