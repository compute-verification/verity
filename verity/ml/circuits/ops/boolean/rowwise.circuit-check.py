"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.rowwise`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The softmax max and both halves of its reciprocal at CMP-92's longest row, N = 32 (its context), and at N = 1, a
    request's first position; one softmax output; and the embedding's coordinate at the byte vocabulary, V = 256 (both
    phases'), and at a vocabulary that isn't a power of two."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import rowwise as BR

    return ([bind(BR.SoftmaxMax, N=n) for n in (1, 32)] + [bind(BR.SoftmaxRecip, N=n, HALF=h) for n in (1, 32) for h in (0, 1)]
            + [BR.SoftmaxOut, bind(BR.EmbeddingCoordinate, V=256), bind(BR.EmbeddingCoordinate, V=200)])
