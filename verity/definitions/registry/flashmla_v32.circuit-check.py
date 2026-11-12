"""circuit-check's bindings of `verity.definitions.registry.flashmla_v32`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at GLM-5.3's statics on the H200 (`flashmla_v32.FLASHMLA_GLM53`): the row, a head over two
    blocks in one split (which the decode kernel writes itself) and over three in two (through the combine), and one head's
    token over one block.  Through them the registered block, split and combine are reached at this instance's widths (the
    block and the split at D = 576, the combine at D = 512).  A token's 8 heads over its 32 blocks at a step's SPLITS (over
    `targets.MAX_GATES`: the twin's bindings stand for the family) are checked by id."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import flashmla_v32 as V

    S = V.FLASHMLA_GLM53
    hd = {"DK": S["DN"] + S["DR"], "DV": S["DN"], "BN": S["BN"], "DOT": S["DOT"], "SCALE": S["SCALE"]}
    return [
        ("FlashMlaV32KvRow_v1", lambda: bind(V.FlashMlaV32KvRow, DN=S["DN"], DR=S["DR"], QT=S["QT"])),
        ("FlashMlaV32DecodeHead_v1:1", lambda: bind(V.FlashMlaV32DecodeHead, SPLITS=(0, 2), **hd)),
        ("FlashMlaV32DecodeHead_v1:2", lambda: bind(V.FlashMlaV32DecodeHead, SPLITS=(0, 1, 3), **hd)),
        ("FlashMlaV32Decode_v1", lambda: bind(V.FlashMlaV32Decode, **{**S, "NH": 1, "SPLITS": (0, 1)})),
    ]
