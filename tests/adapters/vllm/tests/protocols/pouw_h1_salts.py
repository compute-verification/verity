"""Seeded salts for H1 tests whose two runs must commit the same roots (`Seeded`): a test's stand-in for the run's fresh
salts, never the prover's."""
from __future__ import annotations

import hashlib

from verity.protocols.resource_accounting.compute.pouw.prover.serving.h1 import prover as PV


class Seeded(PV.RunSalts):
    """A run's salts drawn from SHAKE256 of ``seed`` and the nonce, whatever order the run draws in: a row's salt is the
    same at a nonce however many rows were drawn before."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed

    def at(self, nonce: int, n: int) -> bytes:
        have = self._by.get(int(nonce), b"")
        if len(have) < 192 * n:
            whole = hashlib.shake_256(f"{self.seed}/{int(nonce)}".encode()).digest(192 * n)
            self._by[int(nonce)] = have + whole[len(have):]
        return super().at(nonce, n)
