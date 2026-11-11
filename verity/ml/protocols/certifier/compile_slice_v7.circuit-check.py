"""circuit-check's bindings of `verity.ml.protocols.certifier.compile_slice_v7`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_compile_slice_v7_roots() -> list[Any]:
    """The compile slice charging the length terms at CompileSlice_v1's demonstration classes (`compile_slice_v7.demo`,
    `demo_async`): CompileSlice_v1's units with ChargeLink_v2{D, 0, 0, 3} in place of ChargeLink_v1{D}, five and two
    decisions and a length slot for each of three arrival slots. Each also with its served rows in blocks of G = 2 steps
    (`CompileSliceRowGroups_v2`)."""
    from verity.core.protocols.certifier.compile_slice import with_block
    from verity.ml.protocols.certifier.compile_slice_v7 import demo, demo_async, unit

    return [unit(**st) for st in (demo()[0], demo_async()[0], with_block(demo()[0], 2), with_block(demo_async()[0], 2))]
