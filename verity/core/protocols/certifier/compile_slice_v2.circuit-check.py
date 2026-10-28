"""circuit-check's bindings of `verity.core.protocols.certifier.compile_slice_v2`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_compile_slice_pool_roots() -> list[Any]:
    """The compile slice with vLLM's block pool at its demonstration class (`compile_slice_v2.demo`): five steps at
    (R 2, W 1, A 2) of 4-token blocks on six blocks, three arrivals, five decisions, a 5-entry pool of 4-bit tokens with
    two copies an entry and 6-bit stamps, 4 units a request; the same with its served rows one step a block (G = 1);
    and at its asynchronous one (`compile_slice_v2.demo_async`). The corpus engines' classes (159 entries, 16 slots,
    lookups of up to 160 units) are about 300M ANDs a step, past what circuit-check traces."""
    from verity.core.protocols.certifier.compile_slice_v2 import demo, demo_async, unit

    return [unit(**demo()[0]), unit(**{**demo()[0], "G": 1}), unit(**demo_async()[0])]
