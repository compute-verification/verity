"""circuit-check's bindings of `verity_flock.rec_cap`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _rec_roots() -> list[Any]:
    """V*'s hold of each Ligerito level's committed tops (`verity_flock.rec_cap`): its one Definition, a held row's words past
    the node then zero words, no AND, the same at every level. It has no parameters; the cap's level and node count are the
    statement's and the population's, not the Definition's."""
    from verity_flock.rec_cap import definition

    return [
        definition(),
    ]
