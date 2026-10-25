"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.softmax_row`'s Definition, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The row softmax at CMP-92's longest row, N = 32 (its context, in both phases), and at N = 1, a request's first
    position."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import softmax_row as SR

    return [bind(SR.SoftmaxRow, N=n) for n in (1, 32)]
