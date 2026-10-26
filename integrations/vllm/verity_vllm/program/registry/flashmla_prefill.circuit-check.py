"""circuit-check's bindings of `verity_vllm.program.registry.flashmla_prefill`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at DeepSeek-V4-Pro's statics on the H200 (`flashmla_prefill.FLASHMLA_PREFILL_DSV4`): the gather's
    row, a token's first and later pairs, a head over two pairs, and one head's prefill over one pair.  A token's 16 heads over
    one pair and over a full C4 row's nine (over `targets.MAX_GATES`: the twin's bindings stand for the family) are checked by
    id."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import flashmla_prefill as P

    S = P.FLASHMLA_PREFILL_DSV4
    blk = {"D": S["D"], "BN": S["BN"], "DOT": S["DOT"], "SCALE": S["SCALE"]}
    return [
        ("FlashMlaGatherRow_v1", lambda: bind(P.FlashMlaGatherRow, **P.FLASHMLA_GATHER_DSV4)),
        ("FlashMlaPrefillPair_v1:1", lambda: bind(P.FlashMlaPrefillPair, FIRST=True, **blk)),
        ("FlashMlaPrefillPair_v1:2", lambda: bind(P.FlashMlaPrefillPair, FIRST=False, **blk)),
        ("FlashMlaPrefillHead_v1", lambda: bind(P.FlashMlaPrefillHead, NP=2, **blk)),
        ("FlashMlaPrefill_v1", lambda: bind(P.FlashMlaPrefill, NH=1, NP=1, **blk)),
    ]
