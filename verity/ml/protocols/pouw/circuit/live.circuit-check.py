"""circuit-check's bindings of `verity.ml.protocols.pouw.circuit.live`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwLiveSlots_v1` and `PouwLiveSlots_v2` at T = 32 and T = 96, two tables' leaf counts (the pouw setting ``rows``, a
    multiple of 32; 96's window slot is 128, so T < M there): the m ≤ T test, T request liveness, index and prefix tests,
    the popcount N, T slot tests against N and m, and each request's slot count against what it takes in round s; v2 adds
    each leaf's liveness against its public cap and a live header's round 0 and cap against it. It grows as T², the
    counts. `PouwLiveRows_v1` at T = 16 with pieces of R = 4 and at T = 8192 with R = 64 (a V3 prefill round's, 128
    pieces): R header and cap tests, R row tests against N, m and base, and each row's interval test against the request
    it names; it grows as R², not with T."""
    from verity.ml.circuits.defs import bind
    from verity.ml.protocols.pouw.circuit import live as LV

    return [*(bind(d, T=t) for d in (LV.PouwLiveSlots, LV.PouwLiveSlotsV2) for t in (32, 96)),
            *(bind(LV.PouwLiveRows, T=t, R=r) for t, r in ((16, 4), (8192, 64)))]
