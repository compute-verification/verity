"""circuit-check's bindings of `verity_vllm.program.registry.boolean_flashmla`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at the smallest statics with every arm (D = 16, one 16-row wgmma block; the circuits are flat, a
    head about a million ANDs): both blocks, both split ends, the combine, a head of two splits, and a decode token over a window
    block and an extra block."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_flashmla as BF

    blk = {"D": 16, "BN": 16, "DOT": BF.FLASHMLA_DSV4["DOT"], "SCALE": 0.25}
    return [
        ("FlashMlaKvRow_v2", lambda: bind(BF.FlashMlaKvRowV2, DN=16, DR=16, QT=8)),
        ("FlashMlaBlock_v2", lambda: bind(BF.FlashMlaBlockV2, FIRST=True, **blk)),
        ("FlashMlaBlock_v2", lambda: bind(BF.FlashMlaBlockV2, FIRST=False, **blk)),
        ("FlashMlaSplit_v2", lambda: bind(BF.FlashMlaSplitV2, NB=1, PARTIAL=False, **blk)),
        ("FlashMlaSplit_v2", lambda: bind(BF.FlashMlaSplitV2, NB=1, PARTIAL=True, **blk)),
        ("FlashMlaCombine_v2", lambda: bind(BF.FlashMlaCombineV2, NS=2, D=1)),
        ("FlashMlaDecodeHead_v2", lambda: bind(BF.FlashMlaDecodeHeadV2, SPLITS=(0, 1, 2), **blk)),
        ("FlashMlaDecode_v2", lambda: bind(BF.FlashMlaDecodeV2, NH=1, SPLITS=(0, 2), NW=16, DN=8, DR=8, QT=8, BN=16,
                                           DOT=BF.FLASHMLA_DSV4["DOT"], SCALE=0.25)),
    ]
