"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.ncp2`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`ncp-v2`'s unit templates at k = 64, and the smallest whole call, its weight side and a window of two calls on it, for
    both P's (`ncp-v2`'s word P and `ncp-v2-shift24`); the package's own tests check their template partitions and the Z
    separator."""
    from verity.core.protocols.pouw.circuit import ncp2

    shape = ncp2.Shape(16, 64, 16)
    out = []
    for kind in ("shift24", "word"):
        out += [t for t in ncp2.templates(64, kind).values() if t not in out]
        out += [ncp2.linear(shape, kind), ncp2.weight(shape, kind), ncp2.window(shape, 2, kind)]
    return out
