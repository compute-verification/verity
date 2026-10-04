"""Space: a lower bound on the memory the prover held when challenged, by a timed audit that it still holds an encoding
of its data. Answering a random block within the deadline needs the block at hand, since rebuilding it takes longer;
so the clock must be near the memory, and the challenge comes from the verifier's hardware at the site. One audit
bounds one instant.

    verifier    draws block indices from its secret, checks the answer and its time, and records the bound
    pinger      at the site: challenges the prover, and reports the answer and the seconds it took (its clock's reading)
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
    seconds: float


@dataclass(frozen=True)
class SpaceBound:
    """If `accepted`: the prover held more than `rho` of the `bytes` encoded, except with probability `delta`."""

    bytes: int
    rho: float
    delta: float
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Space:
    """A prover holding a share f of the blocks answers k uniform challenges within the deadline with probability f**k,
    so one that passes held more than rho = delta**(1/k) of them, except with probability delta."""

    seed: bytes
    blocks: int
    k: int  # blocks challenged
    deadline: float  # seconds
    delta: float

    def verifier(self) -> Actor:
        secret = secrets.token_bytes(32)
        indices = tuple(
            int.from_bytes(derive(secret, "verity/space/v0", str(j)), "big") % self.blocks for j in range(self.k)
        )
        timed = yield from receive(Timed, "pinger", Send(Challenge(indices), "pinger"))
        expected = encode(self.seed, self.blocks)
        reasons = (("wrong blocks",) if timed.blocks != tuple(expected[i] for i in indices) else ()) + (
            (f"answered in {timed.seconds}s, over the deadline",) if timed.seconds > self.deadline else ()
        )
        rho = self.delta ** (1 / self.k)
        return (Send(SpaceBound(self.blocks * len(expected[0]), rho, self.delta, not reasons, reasons)),)

    def pinger(self, seconds: float) -> Actor:
        challenge = yield from receive(Challenge, "verifier")
        answer = yield from receive(Answer, "prover", Send(challenge, "prover"))
        return (Send(Timed(answer.blocks, seconds), "verifier"),)

    def prover(self, encoding: Sequence[bytes]) -> Actor:
        challenge = yield from receive(Challenge, "pinger")
        return (Send(Answer(tuple(encoding[i] for i in challenge.indices)), "pinger"),)
