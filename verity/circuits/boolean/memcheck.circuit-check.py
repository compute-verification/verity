"""circuit-check's bindings of `verity.definitions.circuits.boolean.memcheck`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _memcheck_roots() -> list[Any]:
    """The parts at the RAM pool's record layout (tag 2, address 128, value 128, time 28 bits: three chunks a record, the
    top one mostly constant zeros), and the whole check at a toy binding, three accesses and a list of three entries of
    one-chunk records: what the parts' own checks can't reach is its wiring, the chain from R = W = 1, the times 1..N, the
    first entry and each entry's predecessor, and the final comparison."""
    from verity.definitions.circuits.boolean import memcheck as MC
    from verity.circuits.defs import bind

    return [bind(MC.MemAccess, G=2, A=128, V=128, T=28), bind(MC.MemEntry, G=2, A=128, V=128, T=28),
            bind(MC.MemCheck, N=3, M=3, G=2, A=4, V=4, T=3)]
