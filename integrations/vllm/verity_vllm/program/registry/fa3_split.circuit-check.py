"""circuit-check's bindings of `verity_vllm.program.registry.fa3_split`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at small statics, and the split head at two served rows."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import fa3_split, hopper

    fa3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
    return [
        ("AttnBlockFirst_v1", lambda: bind(fa3_split.AttnBlockFirst, D=16, NVIS=8, BN=16, DOT=fa3["DOT"])),
        # T = 40 at BN = 16 in two splits: blocks 1 and 0 (block 0 below MASKED_FROM, unguarded), then the partial block 2
        ("AttentionHeadSplit_v1", lambda: bind(fa3_split.AttentionHeadSplit, T=40, NBM=3, NS=2, D=16, BN=16, MASKED_FROM=1, **fa3)),
        # the tile's second split is past the row (T = 17 of NBM = 4 blocks): one split combined
        ("AttentionHeadSplit_v1 row past a split", lambda: bind(fa3_split.AttentionHeadSplit, T=17, NBM=4, NS=2, D=16, BN=16,
                                                                MASKED_FROM=1, **fa3)),
        # GLM-5.3's served rows (head 256, the 128 x 80 tile): the last of a 321-token sequence (five one-block splits, the last
        # holding one key) and of a 1000-token one (blocks 6..0 and 12..7, the masked iterations 12 and 11)
        ("AttentionHeadSplit_v1 served 321", lambda: bind(fa3_split.AttentionHeadSplit, T=321, NBM=5, NS=5, D=256, BN=80,
                                                          MASKED_FROM=3, **fa3)),
        ("AttentionHeadSplit_v1 served 1000", lambda: bind(fa3_split.AttentionHeadSplit, T=1000, NBM=13, NS=2, D=256, BN=80,
                                                           MASKED_FROM=11, **fa3)),
        ("AttentionSplit_v1", lambda: bind(fa3_split.AttentionSplit, T=40, NBM=3, NS=2, NH=2, KVH=1, D=16, BN=16, MASKED_FROM=1,
                                           **fa3)),
    ]
