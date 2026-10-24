"""circuit-check's bindings of `verity.ml.definitions.registry.fa2_check_inf`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import b1, fa2_check_inf, hopper

    return [
        ("Attention_v11", lambda: bind(fa2_check_inf.AttentionV11, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1,
                                       DOT=hopper.HopperBF16WgmmaDot16, INV=b1.P.Fa2InvSum)),
    ]
