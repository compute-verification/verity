"""circuit-check's bindings of `verity.ml.definitions.registry.dsa_topk_mask`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

#: GLM-5.3's rows past INDEX_TOPK = K = 2,048 read below their size class (`DsaButterflyRowsPad`): a row of T keys at its class N,
#: the next multiple of K, so N is one of the 63 classes 4,096..131,072 and T one of the N - 2,047..N - 1 keys below it (at T = N
#: the row is `DsaButterflyRows`).  A rule spans each static alone, so its T range is every class's together.  The spots are rows
#: vLLM serves: the first class's shortest (T = K + 1, the most pad) at the latent rows' 576 words, its rows and pad read lazily
#: (`circuit_check.targets.LAZY_OPERANDS`), and at one word its longest (one pad row) and the top class's longest.  At every W it
#: is `DsaButterflyRows{N}`'s one network over the rows and then the pad (``circuit-check --as-call --ranges
#: DsaButterflyRowsPad_v1``)
RANGES = {
    "DsaButterflyRowsPad_v1{T=2049..131071,N=4096..131072 step 2048,K=2048,W=1..576,E=16}": (
        "T=2049,N=4096,W=576", "T=131071,N=131072,W=1",
        "T=4095,N=4096,W=1",
    ),
}


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The mask and the long row over 20 keys selecting 16 (reaching the threshold, slot and key accelerators), and the
    concentrator at small statics shaped as GLM-5.3's (an odd N, K = N - 1; reaching `DsaButterflyRoute_v1` and the tags' scan),
    with picks (N 33, K 8) and in the rows' read (N 7, K 3); the rows' read of 6 rows padded to 9, and of 9 padded to 16, a long
    row's shape (T >= K, N = 2K)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import dsa_topk_mask

    return [
        ("DsaTopKMask_v1", lambda: bind(dsa_topk_mask.DsaTopKMask, N=20, TOPK=16)),
        ("DsaTopKRow_v1", lambda: bind(dsa_topk_mask.DsaTopKRow, N=20, TOPK=16)),
        ("DsaButterflySettings_v1", lambda: bind(dsa_topk_mask.DsaButterflySettings, N=9, K=8)),
        ("DsaButterflySettings_v1-pick", lambda: bind(dsa_topk_mask.DsaButterflySettings, N=33, K=8)),
        ("DsaButterflyRows_v1", lambda: bind(dsa_topk_mask.DsaButterflyRows, N=9, K=8, W=3, E=16)),
        ("DsaButterflyRows_v1-pick", lambda: bind(dsa_topk_mask.DsaButterflyRows, N=7, K=3, W=2, E=16)),
        ("DsaButterflyRowsPad_v1", lambda: bind(dsa_topk_mask.DsaButterflyRowsPad, T=6, N=9, K=8, W=3, E=16)),
        ("DsaButterflyRowsPad_v1-long", lambda: bind(dsa_topk_mask.DsaButterflyRowsPad, T=9, N=16, K=8, W=3, E=16)),
    ]
