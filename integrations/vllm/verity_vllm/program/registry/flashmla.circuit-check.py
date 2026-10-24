"""circuit-check's bindings of `verity_vllm.program.registry.flashmla`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at DeepSeek-V4-Pro's statics on the H200 (`flashmla.FLASHMLA_DSV4`): a split's first and later
    blocks, a split writing bf16 and a partial one, the combine of a lone request's 10 splits, a head over the window and a block
    of compressed rows in two splits, and one head's decode over one window block.  A token's 16 heads (over `targets.MAX_GATES`:
    the twin's bindings stand for the family) are checked by id."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import flashmla as F

    S = F.FLASHMLA_DSV4
    D = S["DN"] + S["DR"]
    blk = {"D": D, "BN": S["BN"], "DOT": S["DOT"], "SCALE": S["SCALE"]}
    return [
        ("FlashMlaKvRow_v1", lambda: bind(F.FlashMlaKvRow, DN=S["DN"], DR=S["DR"], QT=S["QT"])),
        ("FlashMlaBlock_v1:1", lambda: bind(F.FlashMlaBlock, FIRST=True, **blk)),
        ("FlashMlaBlock_v1:2", lambda: bind(F.FlashMlaBlock, FIRST=False, **blk)),
        ("FlashMlaSplit_v1:1", lambda: bind(F.FlashMlaSplit, NB=2, PARTIAL=False, **blk)),
        ("FlashMlaSplit_v1:2", lambda: bind(F.FlashMlaSplit, NB=1, PARTIAL=True, **blk)),
        ("FlashMlaCombine_v1", lambda: bind(F.FlashMlaCombine, NS=10, D=D)),
        ("FlashMlaDecodeHead_v1", lambda: bind(F.FlashMlaDecodeHead, SPLITS=(0, 1, 3), **blk)),
        ("FlashMlaDecode_v1", lambda: bind(F.FlashMlaDecode, **{**S, "NH": 1, "SPLITS": (0, 1), "NW": 64})),
    ]
