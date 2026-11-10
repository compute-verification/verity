"""circuit-check's bindings of `verity.definitions.ops.boolean.argmax`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The bindings `ops.argmax`'s sidecar checks, on bits."""
    from verity.circuits.defs import bind
    from verity.definitions.ops.boolean import argmax as BA

    return ([bind(BA.ArgmaxChunkBest, C=c) for c in (128, 8)] + [bind(BA.ArgmaxChunkIdx, C=c) for c in (128, 8)]
            + [bind(BA.ArgmaxCombine, V=256, C=128), bind(BA.ArgmaxCombine, V=200, C=64), BA.SelectU16, BA.U16Add])
