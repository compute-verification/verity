"""circuit-check's bindings of `verity.core.protocols.certifier.cache_step`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_cache_roots() -> list[Any]:
    """The prefix cache's row pass (`verity.core.protocols.certifier.cache_step`) at a small class that reaches every
    branch (8 entries, units of 4 tokens of 8 bits, 4-bit request numbers, max_model_len 32: 8 units a row), and at
    smallkv's unit and widths (16-token blocks, 17-bit Llama 3 ids, 10-bit request numbers) with 8 entries and 4 units
    a row, within the lowering's budget."""
    from verity.core.protocols.certifier.cache_step import statics, unit

    return [unit(**statics(8, 4, 8, 4, 32)), unit(**statics(8, 16, 17, 10, 64))]
