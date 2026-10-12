"""circuit-check's bindings of `verity_vllm.program.registry.sparse_mla`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import sparse_mla

    return [
        # its sparse MLA (FA3 over the selected rows) at the served statics, two blocks; on bits at DR = DV = BN = 16
        ("SparseMlaBlock_v1", lambda: bind(sparse_mla.SparseMlaBlock, NVIS=64, FIRST=False,
                                           **{k: sparse_mla.SPARSE_MLA_GLM53[k] for k in ("DR", "DV", "BN", "DOT", "SCALE")})),
        ("SparseMlaHead_v1", lambda: bind(sparse_mla.SparseMlaHead, T=65,
                                          **{k: v for k, v in sparse_mla.SPARSE_MLA_GLM53.items() if k != "NH"})),
        ("SparseMla_v1", lambda: bind(sparse_mla.SparseMla, T=65, **sparse_mla.SPARSE_MLA_GLM53)),
    ]
