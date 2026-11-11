"""circuit-check's bindings of `verity.definitions.registry.merge_attn_states`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at small statics."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import merge_attn_states

    return [
        # GLM-5.3's dense-MHA prefill merge (forward_mha's context and suffix) at two heads of one 128-bit pack each; the served
        # 8 heads of 256 are the same family at NH=8, D=256, which `circuit-check` takes on demand
        ("MergeAttnStates_v1", lambda: bind(merge_attn_states.MergeAttnStates, NH=2, D=8)),
    ]
