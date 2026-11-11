"""circuit-check's bindings of `verity.definitions.registry.boolean_sparse_mla`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _glm53_served_roots() -> list[Any]:
    """`SparseMlaHeadBlock_v3` at GLM-5.3's served statics: a row's later 64-key block, its first, and a 16-key first block
    (a short row's); and the units it is cut into at P, `SparseMlaHeadScores_v2` and `SparseMlaHeadValues_v2`.  Each is over
    `targets.MAX_GATES`, so it holds a ``digests`` pin, which keeps it in ``--all``; its description is rows of Calls, and its
    Flock block is read off them (`verity_flock.circuit.unit_block`)."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_sparse_mla

    S = {k: boolean_sparse_mla.SPARSE_MLA_GLM53[k] for k in ("DR", "DV", "BN", "DOT", "SCALE")}
    cols = {k: v for k, v in S.items() if k != "SCALE"}
    # the block cut at P: a later 64-key block's scores and a quarter of its O's columns, and the first block's
    units = ((boolean_sparse_mla.SparseMlaHeadScoresV2, S), (boolean_sparse_mla.SparseMlaHeadValuesV2, {**cols, "G": 128, "C": 0}))
    cut = [bind(d, NVIS=64, FIRST=first, **st) for first in (False, True) for d, st in units]
    return [bind(boolean_sparse_mla.SparseMlaHeadBlockV3, NVIS=nvis, FIRST=first, **S)
            for nvis, first in ((64, False), (64, True), (16, True))] + cut


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_sparse_mla

    return [
        ("SparseMlaBlock_v2", lambda: bind(boolean_sparse_mla.SparseMlaBlockV2, DR=16, DV=16, NVIS=5, FIRST=True, BN=16,
                                           DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"], SCALE=0.0625)),
        ("SparseMlaHead_v2", lambda: bind(boolean_sparse_mla.SparseMlaHeadV2, T=5, DR=16, DV=16, BN=16, SCALE=0.0625,
                                          DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"], INV=boolean_sparse_mla.SPARSE_MLA_GLM53["INV"])),
        ("SparseMlaHeadBlock_v2", lambda: bind(boolean_sparse_mla.SparseMlaHeadBlockV2, DR=16, DV=16, NVIS=5, FIRST=True, BN=16,
                                               DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"], SCALE=0.0625)),
        # a partial first block (its P V step is an `AttnDotStep` over zero rows) and a full later one (every step a
        # `SparseMlaDotStep`); the second holds a ``digests`` pin, which keeps it beside its family's smallest
        ("SparseMlaHeadBlock_v3", lambda: bind(boolean_sparse_mla.SparseMlaHeadBlockV3, DR=16, DV=16, NVIS=5, FIRST=True, BN=16,
                                               DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"], SCALE=0.0625)),
        ("SparseMlaHeadBlock_v3/later", lambda: bind(boolean_sparse_mla.SparseMlaHeadBlockV3, DR=16, DV=16, NVIS=16, FIRST=False,
                                                     BN=16, DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"], SCALE=0.0625)),
        # the block cut at P: a partial first block's units (the values' step an `AttnDotStep` over zero rows) and a full
        # later one's (every step a `SparseMlaDotStep`)
        ("SparseMlaHeadScores_v2", lambda: bind(boolean_sparse_mla.SparseMlaHeadScoresV2, DR=16, DV=16, NVIS=5, FIRST=True,
                                                BN=16, DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"], SCALE=0.0625)),
        ("SparseMlaHeadScores_v2/later", lambda: bind(boolean_sparse_mla.SparseMlaHeadScoresV2, DR=16, DV=16, NVIS=16,
                                                      FIRST=False, BN=16, DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"],
                                                      SCALE=0.0625)),
        ("SparseMlaHeadValues_v2", lambda: bind(boolean_sparse_mla.SparseMlaHeadValuesV2, DR=16, DV=16, NVIS=5, FIRST=True,
                                                BN=16, DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"], G=4, C=1)),
        ("SparseMlaHeadValues_v2/later", lambda: bind(boolean_sparse_mla.SparseMlaHeadValuesV2, DR=16, DV=16, NVIS=16,
                                                      FIRST=False, BN=16, DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"],
                                                      G=8, C=1)),
        ("SparseMlaHeadEnd_v2", lambda: bind(boolean_sparse_mla.SparseMlaHeadEndV2, DV=16,
                                             INV=boolean_sparse_mla.SPARSE_MLA_GLM53["INV"])),
        ("SparseMla_v2", lambda: bind(boolean_sparse_mla.SparseMlaV2, T=5, NH=2, DR=16, DV=16, BN=16, SCALE=0.0625,
                                      DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"], INV=boolean_sparse_mla.SPARSE_MLA_GLM53["INV"])),
    ]
