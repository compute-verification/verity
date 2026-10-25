"""circuit-check's bindings of `verity.ml.definitions.registry.moe_pad`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import moe_pad

    return [
        ("LMoeFfnL_v1", lambda: bind(moe_pad.LMoeFfnL, E=4, TOPK=2, VPT=2, H=16, I=16)),
        ("LMoeBlock_v1", lambda: moe_pad.block(4, 2, 2, 16, 16)),
        ("LMoeBlockL_v1", lambda: bind(moe_pad.LMoeBlockL, E=4, TOPK=2, VPT=2, H=16, I=16)),
        ("LMoeFfn_v1", lambda: bind(moe_pad.LMoeFfn, E=4, TOPK=2, VPT=2, H=16, I=16)),
    ]
