"""circuit-check's bindings of `verity.ml.circuits.requirement_units`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The plan unit over 16 registered blocks (a two-row-a-block table from block 8, two gathers) and 16 written values, a
    plan of four; the memory-map unit with two values, one written block of eight slots, over 16 written values and four
    written blocks. Every clause's comparison is there at each binding; a larger one repeats them."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.requirement_units import MemoryMapStep, PlanMatch

    return [bind(PlanMatch, P=4, NR=16, V=16, G=2, TB=8, RPB=2), bind(MemoryMapStep, K=2, C=1, V=16, M=4, N=16, SLOTS=8)]
