"""circuit-check's bindings of `verity_vllm.definitions.registry.spec`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`verity.circuits.tiny`, the vLLM tests'), and the
    leaf families they do not reach, at small statics; the greedy accept at two drafts over four chunks of logits, the last
    one short, and over 8 logits, where it is lowered whole and compared; and `Accept_v1` at GLM-5.3's five drafts."""
    from verity.circuits.defs import bind
    from verity.circuits.tiny import TINY, TINY_NOBIAS

    from verity_vllm.definitions.registry import spec

    long = {**TINY, "MAX_POS": 16}
    return [
        ("Accept_v1-glm53", lambda: bind(spec.Accept, K=5)),
        ("GreedyAccept_v1", lambda: bind(spec.GreedyAccept, K=2, V=200)),
        ("GreedyAccept_v1-lowered", lambda: bind(spec.GreedyAccept, K=2, V=8)),
        ("AttentionN_v1", lambda: bind(spec.AttentionN, T=5, NH=2, KVH=1, D=16, BN=16)),
        ("ServeSpec_v1", lambda: bind(spec.ServeSpec, C=long, CD={**TINY_NOBIAS, "MAX_POS": 16}, LP=3, K=2, A=(2, 2))),
        ("ServeDraft_v1", lambda: bind(spec.ServeDraft, C=long, LP=3, STEPS=2)),
        ("VerifyBranch_v1", lambda: bind(spec.VerifyBranch, C=long, LP=3, K=2)),
    ]


def _by_calls_roots() -> list[Any]:
    """`GreedyAccept_v1` at GLM-5.3's statics (five drafts, 154,880 logits a row), checked by its calls: its six
    `TokenSelect_v4` rows are past what circuit-check builds whole, and each callee is a target of a sidecar."""
    from verity.circuits.defs import bind

    from verity_vllm.definitions.registry import spec

    return [bind(spec.GreedyAccept, K=5, V=154880)]
