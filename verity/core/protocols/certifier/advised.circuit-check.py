"""circuit-check's bindings of `verity.core.protocols.certifier.advised`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_advised_roots() -> list[Any]:
    """N2's advised counts (`verity.core.protocols.certifier.advised`) at the tests' size classes (T = 8, R_in = 2,
    unsalted and salted), which reach every part at its smallest, at the deployed T = 600: R_in = 1, and R_in = 4
    salted, as the served counts' (`served.circuit-check.py`), and at the certifier-serve rehearsal's served links, T =
    60, salted, R_in = 1 and R_in = 2, which
    `benchmarks/network_traces/n2_advised_zk.py --e2e` proves on a served run's advice. The T = 600 roots are over
    circuit-check's whole-Definition AND budget (60,000), so check holds them to the reference and their digest pins
    without lowering them; T = 512 and T = 60 are under it, so their Flock lowering is checked and their AND count
    pinned."""
    from verity.core.protocols.certifier.advised import unit

    return [
        unit(8, 2, 0),
        unit(8, 2, 1),
        unit(60, 1, 1),
        unit(60, 2, 1),
        unit(512, 4, 1),
        unit(600, 1, 0),
        unit(600, 4, 1),
    ]
