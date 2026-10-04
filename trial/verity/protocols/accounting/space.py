"""Space: bits held at an instant. A lower bound on the memory the prover held when challenged, by a timed audit that it
still holds an encoding of the declared data, against the memory of the site's devices.

    verifier    draws block indices from its secret; checks the answer, its time and the site `site_from` reported;
                records space's line
    pinger      at the site: challenges the prover, and reports the answer and its clock's readings when it challenged
                and when the answer came
    prover      answers from the encoding it holds

A prover holding a share f of the blocks answers k uniform challenges within the deadline with probability f**k, so
one that passes held more than rho = delta**(1/k) of them, except with probability delta. Physically, the bound rests
on the deadline: answering within it needs the block at hand, since rebuilding it takes longer, so the clock must be
near the memory, and the challenge comes from the verifier's hardware at the site. One audit bounds one instant.
Capacity is the site's devices' memory (the census's).

Stands in for `verity_pous`: here a block is a hash of the public seed and its index (`encode`), so rebuilding it is
quick and the bound holds only once the codec's encoding replaces it.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.primitives.events import Actor, Send, gather, receive
from verity.primitives.physical import Device, Site, Window
from verity.primitives.randomness import derive
from verity.protocols.accounting.account import line


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
class Space:
    seed: bytes
    blocks: int
    k: int  # blocks challenged
    deadline: float  # seconds
    census: Mapping[str, Device]  # device type -> its facts
    delta: float

    def verifier(self, site_from: str) -> Actor:
        secret = secrets.token_bytes(32)
        indices = tuple(
            int.from_bytes(derive(secret, "verity/space/v0", str(j)), "big") % self.blocks for j in range(self.k)
        )
        got = yield from gather({Timed: "pinger", Site: site_from}, Send(Challenge(indices), "pinger"))
        timed, expected = got[Timed], encode(self.seed, self.blocks)
        memory, reasons = got[Site].supply(self.census, "memory")
        if timed.blocks != tuple(expected[i] for i in indices):
            reasons += ("wrong blocks",)
        if timed.window.seconds > self.deadline:
            reasons += (f"answered in {timed.window.seconds}s, over the deadline",)
        encoded = self.blocks * len(expected[0])
        held = int(encoded * self.delta ** (1 / self.k))
        return (Send(line("space", timed.window, int(memory), encoded, None, held, held, self.delta, reasons)),)

    def pinger(self, window: Window) -> Actor:
        challenge = yield from receive(Challenge, "verifier")
        answer = yield from receive(Answer, "prover", Send(challenge, "prover"))
        return (Send(Timed(answer.blocks, window), "verifier"),)

    def prover(self, encoding: Sequence[bytes]) -> Actor:
        challenge = yield from receive(Challenge, "pinger")
        return (Send(Answer(tuple(encoding[i] for i in challenge.indices)), "pinger"),)
