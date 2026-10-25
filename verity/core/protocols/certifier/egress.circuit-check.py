"""circuit-check's bindings of `verity.core.protocols.certifier.egress`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_egress_roots() -> list[Any]:
    """N2's egress encoder (`verity.core.protocols.certifier.egress`): a bucket at the tests' size class (R = 8 256-byte
    slots, NR = 16 request slots of K = 16 outputs, K a power of two), at the smallest even cell with K not a power of
    two and two ready words, and at the warden's defaults (R = 64, NR = 16, K = 4,096), the unit a link-window has T of;
    and the router at the tests' window (T·R = 12 slots, NR·K = 20 entries, D = 5) and at a power of two of each
    (T·R = 32, NR·K = 256, D = 8); the map at the tests' warden (lanes of r = 2, D = 2) with NR = 16, at lanes of r = 3
    with D = 0 and more request slots than lanes, and at the warden's defaults (R = 64, r = 4, D = 10)."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.egress import EgressBucket, EgressMap, EgressRoute

    return [
        bind(EgressBucket, B=256, R=8, NR=16, K=16),
        bind(EgressBucket, B=148, R=4, NR=17, K=6),
        bind(EgressBucket, B=256, R=64, NR=16, K=4096),
        bind(EgressRoute, NR=4, K=5, T=3, R=4),
        bind(EgressRoute, NR=16, K=16, T=4, R=8),
        bind(EgressMap, R=8, r=2, NR=16, D=2),
        bind(EgressMap, R=6, r=3, NR=4, D=0),
        bind(EgressMap, R=64, r=4, NR=16, D=10),
    ]
