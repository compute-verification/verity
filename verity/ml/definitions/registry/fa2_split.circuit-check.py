"""circuit-check's bindings of `verity.ml.definitions.registry.fa2_split`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at small statics in Llama-3.2-1B's decode tile on sm_120 (head dim 64, the split kernel's 256
    keys a block): T = 300 in three splits, the last one empty, and in two (a full block, then the partial one); eight splits
    combined."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import fa2_split, hopper
    from verity.ml.definitions.registry import prims as P

    fa2 = {"D": 64, "BN": 256, "DOT": hopper.HopperBF16WgmmaDot16, "INV": P.Fa2InvSum}
    return [
        ("Fa2SplitEnd_v1", lambda: bind(fa2_split.Fa2SplitEnd, D=64, INV=P.Fa2InvSum, SCALE=0.125)),
        ("Fa2Combine_v1", lambda: bind(fa2_split.Fa2Combine, NS=8, D=64)),
        ("Fa2AttentionHeadSplit_v1", lambda: bind(fa2_split.Fa2AttentionHeadSplit, T=300, NS=3, MASKED_FROM=1, **fa2)),
        ("Fa2AttentionSplit_v1", lambda: bind(fa2_split.Fa2AttentionSplit, T=300, NS=2, NH=4, KVH=2, MASKED_FROM=1, **fa2)),
    ]
