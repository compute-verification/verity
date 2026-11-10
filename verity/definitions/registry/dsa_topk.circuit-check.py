"""circuit-check's bindings of `verity.definitions.registry.dsa_topk`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

#: GLM-5.3's sparse MLA reads past INDEX_TOPK keys: its latent rows (W = KV_LORA + QK_ROPE = 576 bf16 words) a row's Waksman
#: settings route to its top 2,048, over every key count from the first row past INDEX_TOPK to the longest served row, checked at
#: these spots (`circuit_check.ranges`; ``circuit-check --as-call --ranges DsaWaksmanRows_v1``).  Past 7,281 keys the rows are
#: over circuit-check's input budget and read lazily (`circuit_check.targets.LAZY_OPERANDS`)
RANGES = {
    "DsaWaksmanRows_v1{N=2049..32533,K=2048,W=576,E=16}": (
        # the ends: the first row past INDEX_TOPK, and the longest served row
        "N=2049", "N=32533",
        # an even row, a power of two and one past it, the last row read whole and the first read lazily, and two long rows
        "N=2050", "N=4096", "N=4097", "N=7281", "N=7282", "N=16384", "N=24576",
    ),
}


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The canonical row's check at GLM-5.3's first row past TOPK (2049 keys, TOPK 2048: one primitive over the row), and the
    Waksman read at small statics shaped as GLM-5.3's (an odd N, K = N - 1; reaching `DsaWaksmanSwap_v1`,
    `DsaWaksmanPick_v1` and both `Xor3x*`)."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import dsa_topk

    return [
        ("DsaTopKLongRow_v1", lambda: bind(dsa_topk.DsaTopKLongRow, N=2049, TOPK=2048)),
        ("DsaWaksmanRows_v1", lambda: bind(dsa_topk.DsaWaksmanRows, N=9, K=8, W=3, E=16)),
        ("DsaWaksmanRows_v1-pick", lambda: bind(dsa_topk.DsaWaksmanRows, N=6, K=3, W=2, E=16)),
        ("DsaWaksmanOrder_v1", lambda: bind(dsa_topk.DsaWaksmanOrder, N=9, TOPK=8)),
    ]
