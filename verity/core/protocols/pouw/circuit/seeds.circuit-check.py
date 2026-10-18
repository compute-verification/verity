"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.seeds`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """The seed and a noise block, on words and on bits (two SHA-512 compressions and one): static-free, so they are catalog
    roots anyway, and here so each is checked whatever its size."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pouw.circuit import seeds as SD

    return [bind(SD.PouwSeed), bind(SD.PouwNoiseBlock), bind(SD.PouwSeedBits), bind(SD.PouwNoiseBlockBits)]
