"""circuit-check's bindings of `verity_flock.rec_cap`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _rec_roots() -> list[Any]:
    """V*'s hold of the table's level-0 cap (`verity_flock.rec_cap`): its one Definition, a held row's words past the node then
    zero words, no AND. It has no parameters; the cap's node count is the population's, not the Definition's."""
    from verity_flock.rec_cap import definition

    return [
        definition(),
    ]
