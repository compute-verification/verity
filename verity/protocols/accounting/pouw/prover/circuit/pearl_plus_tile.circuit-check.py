"""circuit-check's bindings of `verity.protocols.accounting.pouw.prover.circuit.pearl_plus_tile`'s Definitions, in
`circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """Pearl+'s -h3 tile unit and its pieces at the smallest cap instance, a 1 × 1 tile of a one-column unit at K = 256
    (two promotion windows, `cap._promoted`) on the H100's record, forming v1 under -h3: a row's admission
    (`PearlPlusAdmit_v1`) and A's salt-dead counts (`PearlPlusSaltDead_v1`) on words and on bits, the core
    (`PearlPlusTileCore_v1`, its σ `_v2`), the digest's hm96 leaf opening at a depth-1 tree (`PearlPlusTileLeaf_v1`), and the
    whole unit (`PearlPlusTile_v1`, `tile`).  The core and the unit are over circuit-check's whole-lowering budget, so they
    keep a circuit digest, not an AND count; their σ Calls are the pinned units (`pearl_c_call.units`)."""
    from verity.circuits.defs import bind

    from verity.protocols.accounting.pouw.prover.circuit import cap, pearl_plus_tile as T
    from verity.protocols.accounting.pouw.prover.schemes import pearl_c as C
    from verity.protocols.accounting.pouw.prover.schemes.pearl_c_device import DEVICES

    k, dev = 256, "h100"
    scheme = C.PearlC(k_min=k, tile=(1, 1), forming="v1", device=DEVICES[dev], hashing="h3")
    rho, rhor = cap._rho(dev, None), cap.row_rho(scheme, k)
    core = dict(K=k, TM=1, TN=1, N=1, RHO=rho, RHOR=rhor, DEV=dev)
    return [bind(T.PearlPlusAdmit, K=k), bind(T.PearlPlusAdmitV2, K=k),
            bind(T.PearlPlusSaltDead, K=k, DEV=dev), bind(T.PearlPlusSaltDeadV2, K=k, DEV=dev),
            bind(T.PearlPlusTileCore, **core), bind(T.PearlPlusTileCoreV2, **core),
            bind(T.PearlPlusTileLeaf, D=1), T.tile(scheme, k=k, tm=1, tn=1, n=1, depth=1)]
