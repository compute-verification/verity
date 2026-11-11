"""circuit-check's bindings of `verity.definitions.registry.dsa_topk_mask_cut`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

#: the mask whole a unit (`dsa_topk_mask_cut.DsaTopKMaskWhole_v1`) at the long row's served statics, TOPK 2,048 up to
#: `WHOLE_MAX`: the first class, the first whose rows alone need 2^27 bits, and the last
RANGES = {
    "DsaTopKMaskWhole_v1{N=4096..40960 step 2048,TOPK=2048}": ("N=4096", "N=40960", "N=24576"),
}


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Units of `DsaTopKMaskCounted_v1{N=14,TOPK=4}`'s cut in blocks of 8 keys (`dsa_topk_mask_cut.plan(14, 4, 8)`: a block of 8
    and a last one of 6): the first block's count at round 0 (reaching `DsaRowValid_v1` and `DsaTopKDigitOneHot_v1{K=0}`) and
    the second's at rounds 1 and 3 (`DsaRowValidFrom_v1`, the prefix compare), each with `DsaTopKPopCount_v1`; the picks of
    rounds 0, 1 and 3 over both blocks' counts (TOPK a constant, then read, and the last round's key and r per block; at M = 2
    each is under `targets.MAX_GATES`); each block's mask unit (reaching `DsaTopKKeyCompare_v1` and `DsaRadixTies_v1`); and the
    mask whole, its 14 bits in a u16 word.  And the picks the cut serves, each round's over 8 blocks' counts at TOPK 2,048
    (`plan(131072, 2048)`), over `MAX_GATES` and each held to its digest pin."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import dsa_topk_mask_cut as X

    return [
        ("DsaTopKDigitPick_v1-served-0", lambda: bind(X.DsaTopKDigitPick, M=8, K=0, TOPK=2048)),
        ("DsaTopKDigitPick_v1-served-1", lambda: bind(X.DsaTopKDigitPick, M=8, K=1, TOPK=2048)),
        ("DsaTopKDigitPick_v1-served-2", lambda: bind(X.DsaTopKDigitPick, M=8, K=2, TOPK=2048)),
        ("DsaTopKDigitPick_v1-served-3", lambda: bind(X.DsaTopKDigitPick, M=8, K=3, TOPK=2048)),
        ("DsaTopKDigitCount_v1", lambda: bind(X.DsaTopKDigitCount, N=8, AT=0, K=0)),
        ("DsaTopKDigitCount_v1-at", lambda: bind(X.DsaTopKDigitCount, N=6, AT=8, K=1)),
        ("DsaTopKDigitCount_v1-last", lambda: bind(X.DsaTopKDigitCount, N=6, AT=8, K=3)),
        ("DsaTopKDigitPick_v1", lambda: bind(X.DsaTopKDigitPick, M=2, K=0, TOPK=4)),
        ("DsaTopKDigitPick_v1-read", lambda: bind(X.DsaTopKDigitPick, M=2, K=1, TOPK=4)),
        ("DsaTopKDigitPick_v1-last", lambda: bind(X.DsaTopKDigitPick, M=2, K=3, TOPK=4)),
        ("DsaTopKMaskBlock_v1", lambda: bind(X.DsaTopKMaskBlock, N=8, AT=0, TOPK=4)),
        ("DsaTopKMaskBlock_v1-at", lambda: bind(X.DsaTopKMaskBlock, N=6, AT=8, TOPK=4)),
        ("DsaTopKMaskWhole_v1", lambda: bind(X.DsaTopKMaskWhole, N=14, TOPK=4)),
    ]


def _mask_calls_roots() -> list[Any]:
    """The mask units as Calls (their ties `DsaRadixTies_v2`, their valid bits `DsaRowValid_v2`, the whole mask's passes
    `DsaRadixPass_v2`), at the `_v1` units' bindings: each block's mask unit (the second reaching `DsaRowValidOffset_v1`) and
    the mask whole.  And the units as served at TOPK 2,048, each held to its digest pin: a block of 16,384 keys, the row's
    first and one past its start, and the longest row masked whole, 40,960 keys."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import dsa_topk_mask_cut as X

    return [bind(X.DsaTopKMaskBlockV2, N=8, AT=0, TOPK=4), bind(X.DsaTopKMaskBlockV2, N=6, AT=8, TOPK=4),
            bind(X.DsaTopKMaskWholeV2, N=14, TOPK=4), bind(X.DsaTopKMaskBlockV2, N=X.BLOCK, AT=0, TOPK=2048),
            bind(X.DsaTopKMaskBlockV2, N=X.BLOCK, AT=X.BLOCK, TOPK=2048), bind(X.DsaTopKMaskWholeV2, N=X.WHOLE_MAX, TOPK=2048)]


def _pick_calls_roots() -> list[Any]:
    """The picks as Calls (`DsaTopKDigitPick_v2`), at the `_v1` picks' bindings: rounds 0, 1 and 3 over two blocks' counts
    (reaching each of its pieces), and each round's over 8 blocks' counts at TOPK 2,048, as the cut serves them, over
    `targets.MAX_GATES` and each held to its digest pin."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import dsa_topk_mask_cut as X

    return [*(bind(X.DsaTopKDigitPickV2, M=2, K=k, TOPK=4) for k in (0, 1, 3)),
            *(bind(X.DsaTopKDigitPickV2, M=8, K=k, TOPK=2048) for k in range(X.ROUNDS))]
