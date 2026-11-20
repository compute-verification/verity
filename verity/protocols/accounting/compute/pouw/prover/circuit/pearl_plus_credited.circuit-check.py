"""circuit-check's bindings of `verity.protocols.accounting.compute.pouw.prover.circuit.pearl_plus_credited`'s
Definitions, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """The tile on C's words, its noise derived inside (`PearlPlusTileCoreWords_v5`, held as its cut, `CUTS`), and its
    core on words with that noise by wire (`PearlPlusTileCoreWired_v2`), at `pearl_plus_tile`'s core instances: a 1 × 1
    tile of a one-column linear at K = 256 on the H100's record, forming v1 under -h3, the 2 × 1 tile whose second row is
    filler (TR = 1), a column tile of a two-column linear (N = 2) and the 1 × 1 tile on the sm_120 record."""
    from verity.circuits.defs import bind
    from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC

    scheme = "pearl-c-h100-v1-h3"
    nodes = [PC.credited_product(scheme, 256, 1, 1, 1), PC.credited_product(scheme, 256, 2, 1, 1, 1),
             PC.credited_product(scheme, 256, 1, 1, 2), PC.credited_product("pearl-c-sm120-v1-h3", 256, 1, 1, 1)]
    return nodes + [bind(PC.PearlPlusTileCoreWired, **n.bindings) for n in nodes]
