"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_flashmla`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """One binding per family at the smallest statics (D = 16, one 16-row wgmma block): the circuits are flat and every one is over
    `targets.MAX_GATES`, so `select` would keep only a family's smallest, and the suite builds each root (a block is about half a
    million ANDs, a head of two splits 1.4 million).  The block that rescales, the split that writes its partial sums, the
    combine, a head and a decode token of one split each; the other arms (the first block, a split's own output, a head of two
    splits, a token over a window and an extra block) are checked by id."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_flashmla as BF

    blk = {"D": 16, "BN": 16, "DOT": BF.FLASHMLA_DSV4["DOT"], "SCALE": 0.25}
    return [
        ("FlashMlaKvRow_v2", lambda: bind(BF.FlashMlaKvRowV2, DN=16, DR=16, QT=8)),
        ("FlashMlaBlock_v2", lambda: bind(BF.FlashMlaBlockV2, FIRST=False, **blk)),
        ("FlashMlaSplit_v2", lambda: bind(BF.FlashMlaSplitV2, NB=1, PARTIAL=True, **blk)),
        ("FlashMlaCombine_v2", lambda: bind(BF.FlashMlaCombineV2, NS=2, D=1)),
        ("FlashMlaDecodeHead_v2", lambda: bind(BF.FlashMlaDecodeHeadV2, SPLITS=(0, 1), **blk)),
        ("FlashMlaDecode_v2", lambda: bind(BF.FlashMlaDecodeV2, NH=1, SPLITS=(0, 1), NW=16, DN=8, DR=8, QT=8, BN=16,
                                           DOT=BF.FLASHMLA_DSV4["DOT"], SCALE=0.25)),
    ]
