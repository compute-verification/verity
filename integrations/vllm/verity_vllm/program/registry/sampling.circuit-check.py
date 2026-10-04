"""circuit-check's bindings of `verity_vllm.program.registry.sampling`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ir.defs import bind
    from verity_vllm.program.registry import sampling

    return [
        ("GumbelTokenSelect_v1", lambda: bind(sampling.GumbelTokenSelect, V=16)),
        ("GumbelTokenSelectSharedGreedy_v1", lambda: bind(sampling.GumbelTokenSelectSharedGreedy, V=16)),
        ("GumbelTopPTokenSelectSharedGreedy_v1", lambda: bind(sampling.GumbelTopPTokenSelectSharedGreedy, V=16)),
    ]
