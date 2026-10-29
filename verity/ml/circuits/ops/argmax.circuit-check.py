"""circuit-check's bindings of `verity.ml.circuits.ops.argmax`'s Definitions, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _core_roots() -> list[Any]:
    """The e2e demo's row's chunks and combine, V = 256 in two chunks of 128 (Phase 1's and Phase 2's vocabulary), a ragged row (V =
    200 in chunks of 64, its last of 8), and the index word's select and add."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import argmax as A

    return ([bind(A.ArgmaxChunkBest, C=c) for c in (128, 8)] + [bind(A.ArgmaxChunkIdx, C=c) for c in (128, 8)]
            + [bind(A.ArgmaxCombine, V=256, C=128), bind(A.ArgmaxCombine, V=200, C=64), A.SelectU16, A.U16Add])
