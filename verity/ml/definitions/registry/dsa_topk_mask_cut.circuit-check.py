"""circuit-check's bindings of `verity.ml.definitions.registry.dsa_topk_mask_cut`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Units of `DsaTopKMaskCounted_v1{N=20,TOPK=4}`'s cut in blocks of 8 keys (`dsa_topk_mask_cut.plan(20, 4, 8)`): the first
    block's count at round 0 (reaching `DsaRowValid_v1` and `DsaTopKDigitOneHot_v1{K=0}`) and a later block's at rounds 1 and 3
    (`DsaRowValidFrom_v1`, the prefix compare), each with `DsaTopKPopCount_v1`; the picks of rounds 0, 1 and 3 (TOPK a constant,
    then read, and the last round's key and r per block); a mask unit of the first block and of the last, short one (reaching
    `DsaTopKKeyCompare_v1` and `DsaRadixTies_v1`)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import dsa_topk_mask_cut as X

    return [
        ("DsaTopKDigitCount_v1", lambda: bind(X.DsaTopKDigitCount, N=8, AT=0, K=0)),
        ("DsaTopKDigitCount_v1-at", lambda: bind(X.DsaTopKDigitCount, N=8, AT=8, K=1)),
        ("DsaTopKDigitCount_v1-last", lambda: bind(X.DsaTopKDigitCount, N=4, AT=16, K=3)),
        ("DsaTopKDigitPick_v1", lambda: bind(X.DsaTopKDigitPick, M=3, K=0, TOPK=4)),
        ("DsaTopKDigitPick_v1-read", lambda: bind(X.DsaTopKDigitPick, M=3, K=1, TOPK=4)),
        ("DsaTopKDigitPick_v1-last", lambda: bind(X.DsaTopKDigitPick, M=3, K=3, TOPK=4)),
        ("DsaTopKMaskBlock_v1", lambda: bind(X.DsaTopKMaskBlock, N=8, AT=0, TOPK=4)),
        ("DsaTopKMaskBlock_v1-at", lambda: bind(X.DsaTopKMaskBlock, N=4, AT=16, TOPK=4)),
    ]
