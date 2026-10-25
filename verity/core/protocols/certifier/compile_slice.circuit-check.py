"""circuit-check's bindings of `verity.core.protocols.certifier.compile_slice`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_compile_slice_roots() -> list[Any]:
    """The compile slice at its demonstration class (`compile_slice.demo`): four steps at (R 2, W 1, A 2) of 4-token
    blocks, three arrivals, five decisions charged by ChargeLink_v1{5}, a 6-entry prefix cache of 4-bit tokens and 4
    units a request; and at its asynchronous one (`compile_slice.demo_async`): six units at (R 2, W 1, A 2) under async
    scheduling, no prefix cache, the served rows held by request through ``wit``'s map. Each also with its served rows
    in blocks of G = 2 steps (`CompileSliceBlocks_v1`: two ``srv`` rows on the demo, three on the async one). The corpus
    slice's class (`tests/fixtures/compile_slice_smallkv.json`) is about 7.6M ANDs, past what circuit-check traces."""
    from verity.core.protocols.certifier.compile_slice import demo, demo_async, unit, with_block

    return [unit(**st) for st in (demo()[0], demo_async()[0], with_block(demo()[0], 2), with_block(demo_async()[0], 2))]
