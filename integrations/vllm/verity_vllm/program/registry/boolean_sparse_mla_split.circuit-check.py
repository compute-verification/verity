"""circuit-check's bindings of `verity_vllm.program.registry.boolean_sparse_mla_split`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_sparse_mla_split as BX

    dot, inv = BX.SPARSE_MLA_GLM53["DOT"], BX.SPARSE_MLA_GLM53["INV"]
    return [
        ("SparseMlaSplitEnd_v2", lambda: bind(BX.SparseMlaSplitEndV2, DV=16, INV=inv, SCALE=0.0625)),
        ("SparseMlaCombine_v2", lambda: bind(BX.SparseMlaCombineV2, NSPLIT=3, DV=16, INV=inv)),
        # two splits of one block each (T = 17 at BN = 16: a full block, then the partial one)
        ("SparseMlaHeadSplit_v2", lambda: bind(BX.SparseMlaHeadSplitV2, T=17, NS=2, DR=16, DV=16, BN=16, DOT=dot, INV=inv,
                                               SCALE=0.0625)),
        ("SparseMlaSplit_v2", lambda: bind(BX.SparseMlaSplitV2, T=17, NS=2, NH=2, DR=16, DV=16, BN=16, DOT=dot, INV=inv,
                                           SCALE=0.0625)),
    ]
