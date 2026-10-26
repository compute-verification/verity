"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_tp_allreduce`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The TP8 all-reduces on bits: FlashInfer's MNNVL one of 8 elements, one- and two-shot, and vLLM's custom one of 64
    elements, 1-stage and 2-stage (every rank owns one pack, so every START is reached), and torch's multimem one of 128
    elements (16 per owner) at a chain for owners 0-3 and a balanced tree for 4-7, IEEE and FTZ, and one 16-byte vector of it,
    the one-stage unit, at GLM-5.3's tree with IEEE adds and at the balanced tree with FTZ ones; and NCCL's tree one of 8
    elements at WORLD 4, down the TP4 capture's chain and down another."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_tp_allreduce as BA
    from verity.ml.definitions.registry import tp_allreduce as TA

    balanced = (((0, 1), (2, 3)), ((4, 5), (6, 7)))
    trees = (TA.chain(8),) * 4 + (balanced,) * 4
    return [
        *[bind(BA.AllReduceMnnvl, WORLD=8, N=8, SHOT=s) for s in (1, 2)],
        *[bind(BA.AllReduceCustom, WORLD=8, SIZE=64, STAGE=s) for s in (1, 2)],
        *[bind(BA.AllReduceMultimem, WORLD=8, SIZE=128, TREES=trees, FTZ=f) for f in (False, True)],
        *[bind(BA.AllReduceMultimemVector, WORLD=8, TREE=t, FTZ=f) for t, f in ((TA.GLM53_H200_TREE, False), (balanced, True))],
        *[bind(BA.AllReduceNcclTree, WORLD=4, SIZE=8, CHAIN=c) for c in ((0, 1, 2, 3), (2, 0, 3, 1))],
    ]
