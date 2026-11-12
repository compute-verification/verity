"""circuit-check's bindings of `verity.protocols.certifier.prover.lengths_row`'s Definition, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_lengths_row_roots() -> list[Any]:
    """The lengths row's proof block (`verity.protocols.certifier.prover.lengths_row`: a row of the network certifier's
    lengths table computed from the request row and the fails row) at the lengths tests' schedule (two slots a row of
    512-byte cells, u32 places, 8-bit counts); at 4,096-byte cells with u32 places and 18-bit counts, one slot and
    sixteen, whose difference is fifteen slots' gates; and at the two cell sizes whose straddle gates those lack: 505
    bytes, where a later cell's ids leave one byte, so the division's remainder is read, and 509, where the first
    cell's do."""
    from verity.circuits.defs import bind
    from verity.protocols.certifier.prover.lengths_row import LengthsRow

    return [
        bind(LengthsRow, K=2, B=512, IW=32, LW=8),
        bind(LengthsRow, K=1, B=4096, IW=32, LW=18),
        bind(LengthsRow, K=16, B=4096, IW=32, LW=18),
        bind(LengthsRow, K=3, B=505, IW=7, LW=10),
        bind(LengthsRow, K=1, B=509, IW=32, LW=12),
    ]
