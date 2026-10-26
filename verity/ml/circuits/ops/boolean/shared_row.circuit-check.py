"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.shared_row`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The check at a 1 KiB row (G = 512) in segments of 1, 32 and 64 words, at G = 128 of 1 and 64, and at G = 1, 3 and 12 of
    1, 1 and 4 (no tree, and segment counts that aren't powers of two); the wrapper at G = 512 on the s8 coordinate on two
    rows at K = 64 (and at G = 128), on the embedding's coordinate at the frozen vocabulary, V = 256, and, for segments of
    more than one word, on PV's row at D = 4 and the padded softmax row at T = 2. The served PV and softmax rows (D = 64,
    T = 32) are over `targets.MAX_GATES` in the wrapper as they are bare, so a family member this size stands for them."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.circuits.ops.boolean import rowwise as BR
    from verity.ml.circuits.ops.boolean import shared_row as SR
    from verity.ml.circuits.ops.boolean import softmax_row as SM

    s8 = bind(BG.ScaledMmS8CoordinateRows, K=64, R=2, **BG.GEMM_SM90_S8)
    return [
        *(bind(SR.SharedRowCheck, G=G, W=W) for G, W in ((512, 1), (512, 32), (512, 64), (128, 1), (128, 64), (1, 1), (3, 1),
                                                         (12, 4))),
        bind(SR.SharedRowWindow, FN=s8, G=512),
        bind(SR.SharedRowWindow, FN=s8, G=128),
        bind(SR.SharedRowWindow, FN=bind(BR.EmbeddingCoordinate, V=256), G=512),
        bind(SR.SharedRowWindow, FN=bind(BG.PvRow, T=1, K=16, D=4, **BG.GEMM_HOPPER), G=512),
        bind(SR.SharedRowWindow, FN=bind(SM.QkSoftmaxRowPad, T=2, D=16, **BG.GEMM_HOPPER), G=512),
    ]
