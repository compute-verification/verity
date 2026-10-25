"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.cap`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """The prototype of Pearl-C's per-tile cap: its row (pc8's with the salt-dead counts), its word and a one-word tile at
    k = 256 (two promotion windows) on sm_120, the split's units (the strip, the word unit, a tile row's digests and sums,
    the tile and its tally) at the same k, and a tile row's digests and sums at the deployed 64 columns (two chunks).  The
    per-row cap (v2): the one-word tile and its tally at ρr = 1/250 on sm_120, the call and check with one row a side, and
    the tally `cap.served` serves on `pearl-c-h100-v1-h3` at its census k = 256 (64 × 64 at n = 6,144, ρr = 4ρ)."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pouw.circuit import cap, pc8
    from verity.core.protocols.pouw.protocol import Shape
    from verity.core.protocols.pouw.schemes import pearl_c as C
    from verity.core.protocols.pouw.schemes.pearl_c_device import DEVICES

    out = [*cap.templates(256, "sm120").values()]
    out += [t for r, t in cap.unit_templates(256, "sm120").items()
            if r in ("strip", "word", "row_digest", "row_tally", "tile", "tally")]
    out += [bind(cap.Pc8CapRowDigestDef, TN=64), bind(cap.Pc8CapRowTallyDef, K=256, TN=64)]
    split = pc8.Shape(1, 256, 1, tile=(1, 1), dev="sm120", hidden=True, cap=True, row_rho=250)
    served = cap.served(C.PearlC(forming="v1", device=DEVICES["h100"], hashing="h3"), Shape(64, 256, 6144))
    return out + [cap.templates(256, "sm120", row_rho=250)["tile"], cap.unit_templates(256, "sm120", row_rho=250)["tally"],
                  pc8.call(split), pc8.check(split),
                  cap.unit_templates(served.k, served.dev, served.tile, served.n, row_rho=served.row_rho)["tally"]]
