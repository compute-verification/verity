"""circuit-check's bindings of `verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_plus_credited`'s
Definitions, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """The core on C's words (`PearlPlusTileCoreWords_v1`) at `pearl_plus_tile`'s core instances: a 1 × 1 tile of a
    one-column linear at K = 256 on the H100's record, forming v1 under -h3, and the 2 × 1 tile whose second row is
    filler (TR = 1)."""
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC

    scheme = "pearl-c-h100-v1-h3"
    return [PC.credited_product(scheme, 256, 1, 1, 1), PC.credited_product(scheme, 256, 2, 1, 1, 1)]
