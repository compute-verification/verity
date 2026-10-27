"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.rowwise`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The softmax max and both halves of its reciprocal at the e2e demo's longest row, N = 32 (its context), and at N = 1, a
    request's first position; one softmax output; the embedding's coordinate at the byte vocabulary, V = 256 (both
    phases'), and at a vocabulary that isn't a power of two; and both halves of each RMSNorm part at EPS = 1e-5:
    a chunk's sum at C = 128 (Phase 1's row and Phase 2's chunk), a group's at G = 64, and rs at Phase 1's N = 128 with
    G = 1 and Phase 2's N = 16,384 with G = 2."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import rowwise as BR

    rms = [bind(BR.RmsSumSqPartHalf, C=128), bind(BR.F32SumPartHalf, G=64),
           bind(BR.RmsNormRecipHalf, N=128, G=1, EPS=1e-5), bind(BR.RmsNormRecipHalf, N=16384, G=2, EPS=1e-5)]
    return ([bind(BR.SoftmaxMax, N=n) for n in (1, 32)] + [bind(BR.SoftmaxRecip, N=n, HALF=h) for n in (1, 32) for h in (0, 1)]
            + [BR.SoftmaxOut, bind(BR.EmbeddingCoordinate, V=256), bind(BR.EmbeddingCoordinate, V=200)]
            + [bind(d, HALF=h) for d in rms for h in (0, 1)])
