"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.seeds`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """The seed, the unit's two seeds and a noise block, on words and on bits (two SHA-512 compressions, four and one), and
    J10-R's two seeds (four) and its block counter (one 64-bit add): static-free, so they are catalog roots anyway, and here
    so each is checked whatever its size."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pouw.circuit import seeds as SD

    return [bind(SD.PouwSeed), bind(SD.PouwUnitSeeds), bind(SD.PouwNoiseBlock), bind(SD.PouwSeedBits),
            bind(SD.PouwUnitSeedsBits), bind(SD.PouwNoiseBlockBits), bind(SD.PouwCallSeeds), bind(SD.PouwCallSeedsBits),
            bind(SD.PouwAdd64), bind(SD.PouwAdd64Bits)]
