"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.quant`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The bindings `ops.quant`'s sidecar checks, on bits."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import quant as BQ

    return ([bind(BQ.QuantRowS8, N=128, RELU=relu) for relu in (True, False)] + [bind(BQ.QuantRowS8, N=512, RELU=True)]
            + [bind(BQ.RowAbsMax, N=128, RELU=True), bind(BQ.RowAbsMax, N=4, RELU=False), bind(BQ.QuantChunkS8, N=128, RELU=True)]
            + [bind(BQ.RowAbsMax, N=n, RELU=True) for n in (16384, 65536)] + [bind(BQ.QuantChunkS8, N=8192, RELU=True)])
