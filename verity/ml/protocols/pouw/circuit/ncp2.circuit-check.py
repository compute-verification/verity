"""circuit-check's bindings of `verity.ml.protocols.pouw.circuit.ncp2`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`ncp-v2`'s unit templates at k = 64, and the smallest whole call, its weight side and a window of two calls on it, for
    both P's (`ncp-v2`'s word P and `ncp-v2-shift24`); the package's own tests check their template partitions and the Z
    separator. Seven of these families take statics (`NcpDigest_v1`, `NcpFormX_v1`, `NcpFormY_v1`, `NcpTile_v1`,
    `NcpWeight_v1`, `NcpLinear_v1`, `NcpWindow_v1`), so they are checked only at the ids bound here: a listing that binds a
    registered family with no statics gets a TypeError for each, not an id. None lowers whole (each is far over the
    whole-lowering budget, so it is lowered as its template's units), so this module has no pins file."""
    from verity.ml.protocols.pouw.circuit import ncp2

    shape = ncp2.Shape(16, 64, 16)
    out = []
    for kind in ("shift24", "word"):
        out += [t for t in ncp2.templates(64, kind).values() if t not in out]
        out += [ncp2.linear(shape, kind), ncp2.weight(shape, kind), ncp2.window(shape, 2, kind)]
    return out
