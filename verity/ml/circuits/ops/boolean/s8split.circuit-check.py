"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.s8split`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """Both halves of a chunk's accumulator at the split's K = 256 and of a group's sum at G = 64 (Phase 2's K = 65,536), and
    the epilogue at Phase 1's K = 512 (G = 2) and Phase 2's two (G = 64 at K = 16,384, G = 4 after the groups at 65,536)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import s8split as SS

    return ([bind(SS.S8AccumulatorPartHalf, K=256, HALF=h) for h in (0, 1)] + [bind(SS.I32SumPartHalf, G=64, HALF=h) for h in (0, 1)]
            + [bind(SS.ScaledMmS8Epilogue, G=g) for g in (2, 4, 64)])
