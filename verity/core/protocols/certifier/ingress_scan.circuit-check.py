"""circuit-check's bindings of `verity.core.protocols.certifier.ingress_scan`'s Definition, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_ingress_scan_roots() -> list[Any]:
    """The scan's ingress bucket unit at the slow test's class: two lanes of two 512-byte cells, requests of at most
    four cells, Programs started every 3 buckets and a 3-bucket ingress bound, which reaches a first and a continuation
    frame in one block, a new and a continuing block and the anchor's wrap. The w63 class (B 512, r 4, lanes 4, C 47,
    EVERY 10, WITHIN 12) is about 1.37M ANDs, 96 s to trace, so it is checked by name, not in every run."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.ingress_scan import IngressScanBucket

    return [bind(IngressScanBucket, B=512, r=2, lanes=2, C=4, IDB=17, EVERY=3, WITHIN=3)]
