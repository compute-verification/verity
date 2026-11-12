"""circuit-check's bindings of `verity.protocols.pouw.prover.circuit.window_statement`'s Definition, in
`circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwWindowStatement_v1` on the 3-bit grid (`window.GRID_BITS`) at P = 2, a call and the filler, and at P = 16, a
    served execution phase's entries bound (`served_commit.ENTRIES`): the two grid points of the public level, then each
    entry row's layout tests, its range against its n_a·n_b, a call's -h3 root against the public one and its pass tiles'
    end, the filler's range against the grid points, and each next entry against the one before. It grows as P, 4,360
    ANDs an entry: 9,024 at P = 2, whose lowering is pinned and compared with the reference, and 70,064 at P = 16, past
    the 65,536 ANDs a whole Definition is lowered within (`checks.Options.lower_max_ands`), so there its partition, its
    dead and redundant gates and its Flock block are checked and its lowering is not."""
    from verity.circuits.defs import bind
    from verity.protocols.pouw.prover.circuit import window_statement as WS

    return [bind(WS.PouwWindowStatement, P=p, S=3) for p in (2, 16)]
