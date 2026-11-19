"""circuit-check's bindings of `verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_plus_tile`'s Definitions, in
`circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """Pearl+'s -h3 tile core and its pieces at the smallest bound instance, a 1 × 1 tile of a one-column linear at K = 256
    (two promotion windows, `inert_bound._promoted`) on the H100's record, forming v1 under -h3: a row's admission
    (`PearlPlusAdmit_v1`) and A's salt-dead counts (`PearlPlusSaltDead_v1`) on words and on bits, the core
    (`PearlPlusTileCore_v3`, the σ of its unregistered word body) and the tile's noise units (`PearlPlusTileSeed_v1`, each
    `PearlPlusTileLine_v2` and `PearlPlusGramRow_v2`, the σ of theirs), there and at K = 32 for two columns of the
    unpromoted sm_120 chain (G = 0); and the core of a 2 × 1 tile whose second
    row is filler (TR = 1), the smallest with both halves of the real-row rule.  The core is over circuit-check's
    whole-lowering budget, so it keeps a circuit digest, not an AND count; its σ Calls are the pinned units
    (`pearl_c_call.units`)."""
    from verity.circuits.defs import bind

    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_tile as T
    from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c as C
    from verity.protocols.resource_accounting.compute.pouw.prover.schemes.pearl_c_device import DEVICES

    k, dev = 256, "h100"
    scheme = C.PearlC(k_min=k, tile=(1, 1), forming="v1", device=DEVICES[dev], hashing="h3")
    filler = C.PearlC(k_min=k, tile=(2, 1), forming="v1", device=DEVICES[dev], hashing="h3")
    return [bind(T.PearlPlusAdmit, K=k), bind(T.PearlPlusAdmitV2, K=k),
            bind(T.PearlPlusSaltDead, K=k, DEV=dev), bind(T.PearlPlusSaltDeadV2, K=k, DEV=dev),
            T.core(scheme, k, 1, 1, 1),
            T.seed(k, 1), T.seed(32, 2), *(T.line(kind) for kind in T.LINES),
            T.gram_row(k, dev), T.gram_row(32, "sm120-unpromoted"),
            T.core(filler, k, 2, 1, 1, tr=1)]
