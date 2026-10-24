"""circuit-check's bindings of `verity_vllm.program.registry.boolean_flashmla_prefill`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """One binding per family at the smallest statics (D = 32, its two halves one k16 step each; one 16-row wgmma block per
    warpgroup): the circuits are flat and every one but the gather's row is over `targets.MAX_GATES`, so `select` would keep only
    a family's smallest, and the suite builds each root.  The pair that rescales, a head and a token of one pair each; the first
    pair and a head of two pairs are checked by id."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_flashmla_prefill as BP

    blk = {"D": 32, "BN": 16, "DOT": BP.FLASHMLA_PREFILL_DSV4["DOT"], "SCALE": 0.25}
    return [
        ("FlashMlaGatherRow_v2", lambda: bind(BP.FlashMlaGatherRowV2, DN=16, DR=16, QT=8)),
        ("FlashMlaPrefillPair_v2", lambda: bind(BP.FlashMlaPrefillPairV2, FIRST=False, **blk)),
        ("FlashMlaPrefillHead_v2", lambda: bind(BP.FlashMlaPrefillHeadV2, NP=1, **blk)),
        ("FlashMlaPrefill_v2", lambda: bind(BP.FlashMlaPrefillV2, NH=1, NP=1, **blk)),
    ]
