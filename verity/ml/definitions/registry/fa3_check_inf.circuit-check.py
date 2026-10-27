"""circuit-check's bindings of `verity.ml.definitions.registry.fa3_check_inf`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics: v20 with a first block of 16 keys below kBlockN 32 over one unguarded
    block, and its head also at a first block of 8 keys (whose V rows `AttnBlockFirst_v1` pads to a k-step)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import fa3_check_inf, hopper

    fa3 = {"D": 16, "BN": 32, "MASKED_FROM": 1, "DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
    return [
        # GLM-5.3's dense-MHA prefill (DsaDenseMha[vllm-dsa]) at its tile's 80-key blocks, one head: row 160 of a prefill, three key
        # blocks of which block 0 is below the 128-row tile's MASKED_FROM (`targets.fa3_masked_from`).  Head 32 (two k16 QK steps)
        # keeps the catalog's cost down; the served head 256 is the same id at D=256, which `circuit-check` takes on demand
        ("Attention_v4", lambda: bind(fa3_check_inf.AttentionV4, T=161, NH=1, KVH=1, D=32, BN=80, MASKED_FROM=1,
                                      DOT=hopper.HopperBF16WgmmaDot16, INV=hopper.Fa3InvSum)),
        ("AttentionHead_v20", lambda: bind(fa3_check_inf.AttentionHeadV20, T=48, **fa3)),
        ("AttentionHead_v20-padded", lambda: bind(fa3_check_inf.AttentionHeadV20, T=40, **fa3)),
        ("Attention_v20", lambda: bind(fa3_check_inf.AttentionV20, T=48, NH=2, KVH=1, **fa3)),
    ]
