"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.cap`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """The prototype of Pearl-C's per-tile cap: its row (pc8's with the salt-dead counts), its word and a one-word tile at
    k = 256 (two promotion windows) on sm_120, the split's units (the strip, the word unit, a tile row's digests and sums,
    the tile and its tally) at the same k, and a tile row's digests and sums at the deployed 64 columns (two chunks)."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pouw.circuit import cap

    out = [*cap.templates(256, "sm120").values()]
    out += [t for r, t in cap.unit_templates(256, "sm120").items()
            if r in ("strip", "word", "row_digest", "row_tally", "tile", "tally")]
    return out + [bind(cap.Pc8CapRowDigestDef, TN=64), bind(cap.Pc8CapRowTallyDef, K=256, TN=64)]
