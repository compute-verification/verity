"""A deterministic challenger for the tests (`verity.core.protocols.pouw.coins`): `challenger(seed)` draws its
coins and hm96 salts from a counter-mode stream of the seed, so every challenger on one seed holds the same coins and
commitment, and a test that audits one epoch several times asks a fresh one each time (a challenger opens a coin once)."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable

from verity.ml.randomness import derive
from verity.core.protocols.pouw.coins import Challenger, CoinOpening

EPOCH = 7
SEED = b"the tests' challenger"


def entropy(seed: bytes) -> Callable[[int], bytes]:
    key, counter = derive(seed, "verity/pouw/test-coins/v1", {}), itertools.count()
    return lambda n: key.stream(next(counter), n)


def challenger(seed: bytes = SEED, epoch: int = EPOCH) -> Challenger:
    return Challenger(epoch, entropy(seed))


COINS = challenger().commitment


@dataclasses.dataclass
class Answering:
    """A challenger that answers every request with one fixed opening, whatever it is asked to open after."""

    opening: CoinOpening

    def open(self, label: str, after: bytes) -> CoinOpening:
        return self.opening
