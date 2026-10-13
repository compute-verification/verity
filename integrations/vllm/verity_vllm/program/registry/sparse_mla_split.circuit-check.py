"""circuit-check's bindings of `verity_vllm.program.registry.sparse_mla_split`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import sparse_mla_split

    glm = sparse_mla_split.M.SPARSE_MLA_GLM53
    head = {k: v for k, v in glm.items() if k != "NH"}
    return [
        ("MufuLg2Ftz_v1", lambda: sparse_mla_split.MufuLg2Ftz),
        ("F32NeZeroFtz_v1", lambda: sparse_mla_split.F32NeZeroFtz),
        ("F32GtZeroFtz_v1", lambda: sparse_mla_split.F32GtZeroFtz),
        # the split path at the served statics: T = 130 in two splits (two blocks, then the partial one), sixteen splits combined
        ("SparseMlaSplitEnd_v1", lambda: bind(sparse_mla_split.SparseMlaSplitEnd, DV=glm["DV"], INV=glm["INV"], SCALE=glm["SCALE"])),
        ("SparseMlaCombine_v1", lambda: bind(sparse_mla_split.SparseMlaCombine, NSPLIT=16, DV=glm["DV"], INV=glm["INV"])),
        ("SparseMlaHeadSplit_v1", lambda: bind(sparse_mla_split.SparseMlaHeadSplit, T=130, NS=2, **head)),
        ("SparseMlaSplit_v1", lambda: bind(sparse_mla_split.SparseMlaSplit, T=130, NS=2, **glm)),
    ]
