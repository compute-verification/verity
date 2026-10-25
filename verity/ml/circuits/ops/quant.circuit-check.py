"""circuit-check's bindings of `verity.ml.circuits.ops.quant`'s Definitions, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _core_roots() -> list[Any]:
    """The amax of the torch loop's row, H = 128, with and without the ReLU, and over four partial maxima; the pair with and
    without it; both scale halves."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import quant as Q

    return ([bind(Q.RowAbsMax, N=128, RELU=relu) for relu in (True, False)] + [bind(Q.RowAbsMax, N=4, RELU=False)]
            + [bind(Q.QuantS8Pair, RELU=relu) for relu in (True, False)] + [bind(Q.QuantS8Scale, HALF=half) for half in (0, 1)])
