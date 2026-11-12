"""circuit-check's bindings of `verity.ml.definitions.registry.sparse_mla_split`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

#: GLM-5.3's sparse MLA decode at its served statics, over every key count its DSA top-k selects (at most 2,048) and every split
#: count serving takes, checked at these spots (`circuit_check.ranges`; ``circuit-check --as-call --ranges SparseMlaSplit_v1``)
RANGES = {
    "SparseMlaSplit_v1{T=1..2048,NS=1..28,NH=8,DR=64,DV=512,BN=64,DOT=HopperBF16WgmmaDot16_v1,INV=Fa3InvSum_v1,SCALE=0.0625}": (
        # the ends: one key unsplit, and the whole top-2,048 unsplit and in 28 splits (16 of 128 keys)
        "T=1,NS=1", "T=2048,NS=1", "T=2048,NS=28",
        # the most-served rows, the top-2,048 in two, three and eleven splits
        "T=2048,NS=2", "T=2048,NS=3", "T=2048,NS=11",
        # a short row in two splits (a block, then 24 keys), a long one unsplit, and one in 25 single-block splits ending in a
        # partial block
        "T=88,NS=2", "T=257,NS=1", "T=1557,NS=25",
    ),
}


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import sparse_mla_split

    glm = sparse_mla_split.M.SPARSE_MLA_GLM53
    head = {k: v for k, v in glm.items() if k != "NH"}
    G = sparse_mla_split.combine_cols(glm["DV"])
    return [
        # the split path at the served statics: T = 130 in two splits (two blocks, then the partial one)
        ("SparseMlaSplitEnd_v1", lambda: bind(sparse_mla_split.SparseMlaSplitEnd, DV=glm["DV"], INV=glm["INV"], SCALE=glm["SCALE"])),
        # units of SparseMlaSplit_v1's cut (`CUTS`) at the served statics: a split's last column block of its end, and the
        # combine of a column block over the most splits a served token takes
        ("SparseMlaSplitEndCols_v1", lambda: bind(sparse_mla_split.SparseMlaSplitEndCols, DV=glm["DV"], G=G, C=glm["DV"] // G - 1,
                                                  INV=glm["INV"], SCALE=glm["SCALE"])),
        # the token's last split at step 16 (T = 272 in five splits of a block): its 16 keys zero-padded to the block
        ("SparseMlaSplitPart_v1", lambda: bind(sparse_mla_split.SparseMlaSplitPart, T=16, **head)),
        ("SparseMlaCombine_v1", lambda: bind(sparse_mla_split.SparseMlaCombine, NSPLIT=28, DV=G, INV=glm["INV"])),
        # with the lse, as the dense-MHA prefill runs it: head 256, the five splits a served row reaches at most
        ("SparseMlaCombineLse_v1", lambda: bind(sparse_mla_split.SparseMlaCombineLse, NSPLIT=5, DV=256, INV=glm["INV"])),
        ("SparseMlaHeadSplit_v1", lambda: bind(sparse_mla_split.SparseMlaHeadSplit, T=130, NS=2, **head)),
        ("SparseMlaSplit_v1", lambda: bind(sparse_mla_split.SparseMlaSplit, T=130, NS=2, **glm)),
    ]
