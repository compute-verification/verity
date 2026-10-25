"""circuit-check's bindings of `verity.ml.circuits.ops.quant`'s Definitions, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _core_roots() -> list[Any]:
    """The row unit at the torch loop's width, H = 128, with and without the ReLU, and at Phase 1's 512; the amax of a row of 128
    and over four partial maxima; a chunk of 128."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import quant as Q

    return ([bind(Q.QuantRowS8, N=128, RELU=relu) for relu in (True, False)] + [bind(Q.QuantRowS8, N=512, RELU=True)]
            + [bind(Q.RowAbsMax, N=128, RELU=True), bind(Q.RowAbsMax, N=4, RELU=False), bind(Q.QuantChunkS8, N=128, RELU=True)])
