"""circuit-check's bindings of `verity.protocols.certifier.prover.cache_step_v3`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pool_stream_roots() -> list[Any]:
    """The pool stream at the certifier test's widths (a 6-block pool, 4-bit requests, 3-bit block indices, 3-bit
    holders, 8-bit times) with two slots and two list entries: each part once, and the wiring the parts' own checks
    can't reach, the two scans' carries from the initial registers and the verdict."""
    from verity.protocols.certifier.prover.cache_step_v3 import statics, unit

    return [unit(**statics(NS=2, NE=2, NB=6, RW=4, KW=3, REFW=3, TW=8))]
