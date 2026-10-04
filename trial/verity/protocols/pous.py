"""Proof of useful space: a timed audit that the prover still holds an encoding of its data. Answering a random block
within the deadline needs the block itself, since rebuilding it takes longer.

    verifier    draws block indices from its secret, checks the answer and its time, and records the bound
    pinger      at the site: challenges the prover, and reports the answer and how long it took (its clock's reading)
    prover      answers from the encoding it holds

Stands in for `verity_pous`: here a block is a hash of the public seed and its index (`encode`), so rebuilding it is
quick and the bound holds only once the codec's encoding replaces it.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Sequence
from dataclasses import dataclass

from verity.primitives.events import Actor, Send, receive
from verity.primitives.randomness import derive


def encode(seed: bytes, blocks: int) -> tuple[bytes, ...]:
    return tuple(hashlib.sha256(seed + i.to_bytes(8, "big")).digest() for i in range(blocks))


@dataclass(frozen=True)
class Challenge:
    indices: tuple[int, ...]


@dataclass(frozen=True)
class Answer:
    blocks: tuple[bytes, ...]


@dataclass(frozen=True)
class Timed:
    """Pinger to verifier: the prover's answer, and the seconds between challenge and answer."""

    blocks: tuple[bytes, ...]
    elapsed: float


@dataclass(frozen=True)
class SpaceBound:
    """If `accepted`: the prover held more than `rho` of the `bytes` encoded, except with probability `delta`."""

    bytes: int
    rho: float
    delta: float
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Pous:
    seed: bytes
    blocks: int
    k: int  # blocks challenged
    deadline: float  # seconds
    rho: float
    delta: float
    name: str = "pous"

    def verifier(self, to: tuple[str, ...] = ()) -> dict[str, Actor]:
        return {f"verifier/{self.name}": self._verifier(to)}

    def pinger(self, elapsed: float) -> dict[str, Actor]:
        return {f"pinger/{self.name}": self._pinger(elapsed)}

    def prover(self, encoding: Sequence[bytes]) -> dict[str, Actor]:
        return {f"prover/{self.name}": self._prover(encoding)}

    def _verifier(self, to: tuple[str, ...]) -> Actor:
        secret = secrets.token_bytes(32)
        indices = tuple(int.from_bytes(derive(secret, "verity/pous/v0", str(j)), "big") % self.blocks for j in range(self.k))
        timed = yield from receive(Timed, "pinger", Send(Challenge(indices), to=(f"pinger/{self.name}",)))
        expected = encode(self.seed, self.blocks)
        reasons = (("wrong blocks",) if timed.blocks != tuple(expected[i] for i in indices) else ()) + (
            (f"answered in {timed.elapsed}s, over the deadline",) if timed.elapsed > self.deadline else ()
        )
        bound = SpaceBound(self.blocks * len(expected[0]), self.rho, self.delta, not reasons, reasons)
        return (Send(bound, to=to),)

    def _pinger(self, elapsed: float) -> Actor:
        challenge = yield from receive(Challenge, "verifier")
        answer = yield from receive(Answer, "prover", Send(challenge, to=(f"prover/{self.name}",)))
        return (Send(Timed(answer.blocks, elapsed), to=(f"verifier/{self.name}",)),)

    def _prover(self, encoding: Sequence[bytes]) -> Actor:
        challenge = yield from receive(Challenge, "pinger")
        return (Send(Answer(tuple(encoding[i] for i in challenge.indices)), to=(f"pinger/{self.name}",)),)
