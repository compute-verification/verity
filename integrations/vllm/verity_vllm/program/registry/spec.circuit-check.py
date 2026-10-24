"""circuit-check's bindings of `verity_vllm.program.registry.spec`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from circuit_check.targets import TINY, TINY_NOBIAS
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import spec

    long = {**TINY, "MAX_POS": 16}
    return [
        ("AttentionN_v1", lambda: bind(spec.AttentionN, T=5, NH=2, KVH=1, D=16, BN=16)),
        ("ServeSpec_v1", lambda: bind(spec.ServeSpec, C=long, CD={**TINY_NOBIAS, "MAX_POS": 16}, LP=3, K=2, A=(2, 2))),
        ("ServeDraft_v1", lambda: bind(spec.ServeDraft, C=long, LP=3, STEPS=2)),
        ("VerifyBranch_v1", lambda: bind(spec.VerifyBranch, C=long, LP=3, K=2)),
    ]
