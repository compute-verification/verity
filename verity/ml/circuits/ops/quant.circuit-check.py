"""circuit-check's bindings of `verity.ml.circuits.ops.quant`'s Definitions, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _core_roots() -> list[Any]:
    """The pair and both scale halves at the torch loop's width, H = 128, with and without the ReLU."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import quant as Q

    return [bind(Q.QuantS8Pair, H=128, RELU=relu) for relu in (True, False)] + [
        bind(Q.QuantS8Scale, H=128, HALF=half, RELU=relu) for half in (0, 1) for relu in (True, False)]
