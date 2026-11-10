"""circuit-check's bindings of `verity.ml.protocols.certifier.cache_step_v2`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_cache_pool_roots() -> list[Any]:
    """The block pool's call (`verity.ml.protocols.certifier.cache_step_v2`) at the certifier test's small class, which
    reaches every branch (8 entries, units of 4 tokens of 3 bits, 4-bit request numbers, max_model_len 24: 6 units a
    call; 3 slots, 2 copies an entry, 7 blocks, 6-bit stamps), and at smallkv's unit and widths (16-token blocks, 17-bit
    Llama 3 ids, 10-bit request numbers) with 8 entries, 4 units a call, 4 slots, one copy an entry and 9 blocks, within
    the lowering's budget."""
    from verity.ml.protocols.certifier.cache_step_v2 import statics, unit

    return [unit(**statics(8, 4, 3, 4, 24, 3, 2, 7, 6)), unit(**statics(8, 16, 17, 10, 64, 4, 1, 9, 8))]
