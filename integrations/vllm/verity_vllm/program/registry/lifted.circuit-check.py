"""circuit-check's bindings of `verity_vllm.program.registry.lifted`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from circuit_check.targets import TINY
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import b1, lifted

    lifted_profile = {"GEMM": b1.GEMM_AMPERE, "ATTENTION": b1.ATTENTION_FA2, "NORM": b1.norm_bundle(TINY)}
    return [
        ("Prefill_v1", lambda: bind(lifted.Prefill, C=TINY, LP=3, **lifted_profile)),
        ("StepBody_v1", lambda: bind(lifted.StepBody, C=TINY, T=4, **lifted_profile)),
        ("LServe_v2", lambda: lifted.lserve(TINY, 3, 2, (7,), lag=0)),
    ]
