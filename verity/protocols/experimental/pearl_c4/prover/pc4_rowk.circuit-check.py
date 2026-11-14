"""circuit-check's bindings of `verity.protocols.experimental.pearl_c4.prover.pc4_rowk`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """Pearl-C4's row units cut along k (``Shape.row_k``) in two segments, at k = 128."""
    from verity.protocols.experimental.pearl_c4.prover import pc4_rowk

    cut = [p for side in ("a", "b") for pieces in (pc4_rowk.pieces4(128, 64, side),)
           for p in (pieces[0], pieces[1], *pieces[2], pieces[3])]
    seen: set[str] = set()
    return [t for t in cut if t.id not in seen and not seen.add(t.id)]
