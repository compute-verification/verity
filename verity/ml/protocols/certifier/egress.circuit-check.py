"""circuit-check's bindings of `verity.ml.protocols.certifier.egress`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_egress_roots() -> list[Any]:
    """N2's egress bucket (`verity.ml.protocols.certifier.egress`, the generator form: the row computed from the
    state), the unit a link-window has T of: at egress_zk.py's
    shape (8 256-byte cells in 4 lanes of 2, 4 request slots of 16 outputs, 18-bit ids, D = 2), at the scan note's
    class (16 256-byte cells in 4 lanes of 4, 4 request slots of 64 outputs, 17-bit ids, D = 22), and at the smallest
    even cell with lanes of 3, more request slots than lanes, K not a power of two and D = 0."""
    from verity.ml.circuits.defs import bind
    from verity.ml.protocols.certifier.egress import EgressBucket

    return [
        bind(EgressBucket, B=256, R=8, r=2, NR=4, K=16, IB=18, D=2),
        bind(EgressBucket, B=256, R=16, r=4, NR=4, K=64, IB=17, D=22),
        bind(EgressBucket, B=148, R=6, r=3, NR=3, K=5, IB=7, D=0),
    ]
