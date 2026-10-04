"""Proof of useful space: a lower bound on the share of an encoding of the declared data the prover held when
challenged, if it answered faster than it could rebuild a block.

    verifier    draws block indices from its secret, checks the answers the pinger relays; records `Held`
    pinger      at the site: challenges the prover, and relays the answer with its clock's readings when it challenged
                and when the answer came
    prover      answers from the encoding it holds

A prover holding a share f of the blocks answers k uniform challenges with the blocks at hand with probability f**k, so
one that passes held more than rho = delta**(1/k) of them, except with probability delta. "At hand" is physical: the
answer came quicker than rebuilding a block takes, which is `reconcile`'s to judge from the pinger's window, since how
long rebuilding takes depends on the devices at the site. One audit bounds one instant.

Stands in for `verity_pous`: here a block is a hash of the public seed and its index (`encode`), so rebuilding it is
quick and the bound holds only once the codec's encoding replaces it.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Sequence
from dataclasses import dataclass

from verity.primitives.events import Actor, Send, receive
from verity.primitives.physical import Window
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
    """Pinger to verifier: the prover's answer, and when the challenge went out and the answer came, by its clock."""

    blocks: tuple[bytes, ...]
    window: Window


@dataclass(frozen=True)
class Held:
    """Of the `encoded` bytes, the prover held at least `held` over `window` if it answered within it faster than it
    could rebuild a block; except with probability `delta`. Nothing, with any reason."""

    window: Window
    encoded: int
    held: int
    delta: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Pous:
    seed: bytes
    blocks: int
    k: int  # blocks challenged
    delta: float

    def verifier(self) -> Actor:
        secret = secrets.token_bytes(32)
        indices = tuple(
            int.from_bytes(derive(secret, "verity/space/v0", str(j)), "big") % self.blocks for j in range(self.k)
        )
        timed = yield from receive(Timed, "pinger", Send(Challenge(indices), "pinger"))
        expected = encode(self.seed, self.blocks)
        reasons = ("wrong blocks",) if timed.blocks != tuple(expected[i] for i in indices) else ()
        encoded = self.blocks * len(expected[0])
        held = int(encoded * self.delta ** (1 / self.k))
        return (Send(Held(timed.window, encoded, held, self.delta, reasons)),)

    def pinger(self, window: Window) -> Actor:
        challenge = yield from receive(Challenge, "verifier")
        answer = yield from receive(Answer, "prover", Send(challenge, "prover"))
        return (Send(Timed(answer.blocks, window), "verifier"),)

    def prover(self, encoding: Sequence[bytes]) -> Actor:
        challenge = yield from receive(Challenge, "pinger")
        return (Send(Answer(tuple(encoding[i] for i in challenge.indices)), "pinger"),)
