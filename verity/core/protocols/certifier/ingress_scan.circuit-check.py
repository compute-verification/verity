"""circuit-check's bindings of `verity.core.protocols.certifier.ingress_scan`'s Definition, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_ingress_scan_roots() -> list[Any]:
    """The scan's ingress bucket unit and its first bucket's at the slow test's class: two lanes of two 512-byte
    cells, requests of at most four cells, a timestep of 2 buckets (two Program slots each), a 3-bucket ingress bound,
    the warden's delay 5, 8 outputs and 100 prompt ids a request, which reaches a first and a continuation frame in one
    block, a new and a continuing block, the anchor's wrap, two requests anchored at one boundary, each slot's
    occupancy word, the outputs written at the anchor and a slot held through its egress. The w63 class (B 512, r 4,
    lanes 4, C 47, EVERY 16, WITHIN 12, D 27, KO 64, KI 1,024) is about 1.2M ANDs, 100 s to trace, so it is checked by
    name, not in every run."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.ingress_scan import IngressScanBucket, IngressScanFirst

    statics = dict(B=512, r=2, lanes=2, C=4, IDB=17, EVERY=2, WITHIN=3, D=5, KO=8, KI=100)
    return [bind(IngressScanBucket, **statics), bind(IngressScanFirst, **statics)]


def _network_ingress_scan_w63k4_roots() -> list[Any]:
    """The two units at the w63k4 class (scan_zk.py's ``w63k4``: w63's at KO 4 and KI 32, a window circuit's of 8
    slots, 32 prompt ids and 4 outputs a request), the circuits its scan registers: their digest pins hold the units
    the deployment proves, which the test class's can't, and they reach four lanes' slot entries, a 47-cell slot and an
    ingress bound below the timestep. As large as w63's in ANDs (a slot's C·K prompt ids dominate the state), so this
    unit is its own, rerun only when what it reads changes."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.ingress_scan import IngressScanBucket, IngressScanFirst

    statics = dict(B=512, r=4, lanes=4, C=47, IDB=17, EVERY=16, WITHIN=12, D=27, KO=4, KI=32)
    return [bind(IngressScanBucket, **statics), bind(IngressScanFirst, **statics)]
