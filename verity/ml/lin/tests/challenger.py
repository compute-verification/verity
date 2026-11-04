"""A test challenger: seeded coins, independent of what they follow, each logged beside the transcript digest it came after."""
from __future__ import annotations

import random

from verity_lin.session import Coin


class SeededChallenger:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self.log: list[Coin] = []

    def flip(self, after: bytes, nbytes: int) -> bytes:
        data = self._rng.randbytes(nbytes)
        self.log.append(Coin(after, data))
        return data


def salts(seed: int):
    rng = random.Random(seed)
    return lambda n: rng.randbytes(n)
