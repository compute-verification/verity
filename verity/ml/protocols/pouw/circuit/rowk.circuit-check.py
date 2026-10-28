"""circuit-check's bindings of `verity.ml.protocols.pouw.circuit.rowk`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """Both tile checks' row units cut along k (``Shape.row_k``) in two segments: Pearl-C4's pieces at k = 128, and Pearl-C's
    at k = 256 on the H100's and sm_120's records and at k = 64 unpromoted."""
    from verity.ml.protocols.pouw.circuit import rowk

    cut = [p for side in ("a", "b") for pieces in (rowk.pieces4(128, 64, side), rowk.pieces8(256, 128, "h100", side),
                                                    rowk.pieces8(256, 128, "sm120", side),
                                                    rowk.pieces8(64, 32, "sm120-unpromoted", side))
           for p in (pieces[0], pieces[1], *pieces[2], pieces[3])]
    seen: set[str] = set()
    return [t for t in cut if t.id not in seen and not seen.add(t.id)]
