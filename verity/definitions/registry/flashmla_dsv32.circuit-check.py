"""circuit-check's bindings of `verity.definitions.registry.flashmla_dsv32`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at GLM-5.3's statics on the H200 (`flashmla_dsv32.FLASHMLA_GLM53`).  Each unit a token is cut
    into (`flashmla_dsv32.CUTS`) at the size serving stages it: the row; a split's first block and a later one; a split's end
    in both arms; the weights and 256 columns of the weighing at 32 splits, the most a step gives a token.  Then the Calls that
    are cut, over a few blocks: a head of two blocks in one split (which the decode kernel writes itself) and of three in two
    (through the combine), and two heads' token over three blocks.  A token's 8 heads over its 32 blocks at a step's SPLITS is
    the same body over more blocks (past `targets.MAX_GATES`) and is checked by id."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import flashmla_dsv32 as V

    S = V.FLASHMLA_GLM53
    hd = {"DK": S["DN"] + S["DR"], "DV": S["DN"], "BN": S["BN"], "DOT": S["DOT"], "SCALE": S["SCALE"]}
    return [
        ("FlashMlaV32KvRow_v1", lambda: bind(V.FlashMlaV32KvRow, DN=S["DN"], DR=S["DR"], QT=S["QT"])),
        ("FlashMlaV32Block_v1:1", lambda: bind(V.FlashMlaV32Block, FIRST=True, **hd)),
        ("FlashMlaV32Block_v1:2", lambda: bind(V.FlashMlaV32Block, FIRST=False, **hd)),
        ("FlashMlaV32SplitEnd_v1:1", lambda: bind(V.FlashMlaV32SplitEnd, DV=S["DN"], PARTIAL=False)),
        ("FlashMlaV32SplitEnd_v1:2", lambda: bind(V.FlashMlaV32SplitEnd, DV=S["DN"], PARTIAL=True)),
        ("FlashMlaV32CombineWeights_v1", lambda: bind(V.FlashMlaV32CombineWeights, NS=32)),
        ("FlashMlaV32CombineValues_v1", lambda: bind(V.FlashMlaV32CombineValues, NS=32, G=V.COMBINE_G)),
        ("FlashMlaV32DecodeHead_v1:1", lambda: bind(V.FlashMlaV32DecodeHead, SPLITS=(0, 2), **hd)),
        ("FlashMlaV32DecodeHead_v1:2", lambda: bind(V.FlashMlaV32DecodeHead, SPLITS=(0, 1, 3), **hd)),
        ("FlashMlaV32Decode_v1", lambda: bind(V.FlashMlaV32Decode, **{**S, "NH": 2, "SPLITS": (0, 1, 3)})),
    ]
