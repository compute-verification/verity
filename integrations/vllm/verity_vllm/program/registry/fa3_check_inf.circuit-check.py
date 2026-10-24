"""circuit-check's bindings of `verity_vllm.program.registry.fa3_check_inf`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import fa3_check_inf, hopper

    return [
        # GLM-5.3's dense-MHA prefill (DsaDenseMha[vllm-dsa]) at its tile's 80-key blocks, one head: row 160 of a prefill, three key
        # blocks of which block 0 is below the 128-row tile's MASKED_FROM (`targets.fa3_masked_from`).  Head 32 (two k16 QK steps)
        # keeps the catalog's cost down; the served head 256 is the same id at D=256, which `circuit-check` takes on demand
        ("Attention_v4", lambda: bind(fa3_check_inf.AttentionV4, T=161, NH=1, KVH=1, D=32, BN=80, MASKED_FROM=1,
                                      DOT=hopper.HopperBF16WgmmaDot16, INV=hopper.Fa3InvSum)),
    ]
