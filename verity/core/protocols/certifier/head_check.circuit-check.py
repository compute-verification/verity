"""circuit-check's bindings of `verity.core.protocols.certifier.head_check`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_roots() -> list[Any]:
    """N2's head check (`verity.core.protocols.certifier.head_check`) for a 9-byte link name ("responses") at T = 4, with
    and without an occupancy profile, which reaches the opening, the fields, the counts and the signed digest (the
    head's record digest and its salted digest) at their smallest; the count check at the deployed T = 600,
    32·T − 1 = 19,199 ANDs; the served link's tests' heads, T = 8 without a profile; the certifier-serve
    rehearsal's size classes, a 6-byte link name ("served") at T = 60 with and without a profile; and the active
    warden's link ("warden") at its schedule's T = 32, which has no profile (`service.WardenProfile`), so their gates
    name a pinned circuit digest and a proof of their unit is accepted."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.certifier.head_check import NetworkHeadCounts, unit

    return [
        unit(9, 4, True),
        unit(9, 4, False),
        bind(NetworkHeadCounts, T=600),
        unit(9, 8, False),
        unit(6, 60, True),
        unit(6, 60, False),
        unit(6, 32, False),
    ]
