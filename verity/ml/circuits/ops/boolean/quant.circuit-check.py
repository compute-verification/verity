"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.quant`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The bindings `ops.quant`'s sidecar checks, on bits."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import quant as BQ

    return ([bind(BQ.RowAbsMax, N=128, RELU=relu) for relu in (True, False)] + [bind(BQ.RowAbsMax, N=4, RELU=False)]
            + [bind(BQ.QuantS8Pair, RELU=relu) for relu in (True, False)] + [bind(BQ.QuantS8Scale, HALF=half) for half in (0, 1)])
