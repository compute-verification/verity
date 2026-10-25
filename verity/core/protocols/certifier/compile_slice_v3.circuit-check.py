"""circuit-check's bindings of `verity.core.protocols.certifier.compile_slice_v3`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_compile_slice_v3_roots() -> list[Any]:
    """The compile slice computing its schedule at CompileSlice_v2's demonstration classes (`compile_slice_v3.demo`,
    `demo_async`): five steps at (R 2, W 1, A 2) of 4-token blocks on six blocks, three arrivals, a 5-entry pool of
    4-bit tokens with two copies an entry and 6-bit stamps, 4 units a request, its advice five decisions and the step
    count (8 bits). At 76,554 and 83,287 ANDs both are past the default whole-lowering budget: their correspondence and
    pins are checked with ``--lower-max-ands 100000``."""
    from verity.core.protocols.certifier.compile_slice_v3 import demo, demo_async, unit

    return [unit(**demo()[0]), unit(**demo_async()[0])]
