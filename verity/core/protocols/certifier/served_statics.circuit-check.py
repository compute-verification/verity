"""circuit-check's bindings of `verity.core.protocols.certifier.served_statics`' Definitions, in `circuit_check.targets'`
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_served_statics_roots() -> list[Any]:
    """The tie at the compile slice's class, corpus-smallkv's window of seven steps at R 2, its engine's L 4096 and T 256 and
    request ids of 6 bits (`verity/core/protocols/certifier/tests/fixtures/compile_slice_smallkv.json`), as one unit (G = P) and as seven (G = 1), with the
    step; and at the sealed served run's class (R 8, IW 16, L 512, T 2048) at G = 16, its ZK run's unit."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.served_statics import ServedStaticsStep, unit

    return [
        bind(ServedStaticsStep, R=2, L=4096, T=256),
        unit(R=2, G=7, IW=6, L=4096, T=256),
        unit(R=2, G=1, IW=6, L=4096, T=256),
        unit(R=8, G=16, IW=16, L=512, T=2048),
    ]
