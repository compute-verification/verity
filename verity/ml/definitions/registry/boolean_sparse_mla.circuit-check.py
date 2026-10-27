"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_sparse_mla`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_sparse_mla

    return [
        ("SparseMlaBlock_v2", lambda: bind(boolean_sparse_mla.SparseMlaBlockV2, DR=16, DV=16, NVIS=5, FIRST=True, BN=16,
                                           DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"], SCALE=0.0625)),
        ("SparseMlaHead_v2", lambda: bind(boolean_sparse_mla.SparseMlaHeadV2, T=5, DR=16, DV=16, BN=16, SCALE=0.0625,
                                          DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"], INV=boolean_sparse_mla.SPARSE_MLA_GLM53["INV"])),
        ("SparseMla_v2", lambda: bind(boolean_sparse_mla.SparseMlaV2, T=5, NH=2, DR=16, DV=16, BN=16, SCALE=0.0625,
                                      DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"], INV=boolean_sparse_mla.SPARSE_MLA_GLM53["INV"])),
    ]
