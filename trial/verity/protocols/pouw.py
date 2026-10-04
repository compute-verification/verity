"""Proof of useful work: the verifier salts the registered work, the prover returns the shares its salted work turned
up, and the verifier bounds the work from the valid ones.

    verifier    takes the registration, sends a salt drawn from its secret, and bounds the work
    prover      searches for shares under the salt

Stands in for `verity_pouw`, whose shares are tiles of the registered matmuls, checked by recomputation. Here a share is
a nonce whose hash with the salt and the registration starts with `difficulty` zero bits, one try per gate the
registered executions ran, so the work bounded is hashing.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

from verity.primitives.events import Actor, Send, receive
from verity.primitives.randomness import derive
from verity.primitives.registration import Registration


@dataclass(frozen=True)
class Salt:
    salt: bytes


@dataclass(frozen=True)
class Shares:
    nonces: tuple[int, ...]


@dataclass(frozen=True)
class UsefulWork:
    """About `work` tries went into the registration's salted work, except with probability `delta`."""

    registration: str  # its digest
    work: int
    delta: float


@dataclass(frozen=True)
class Pouw:
    difficulty: int  # leading zero bits a share's hash needs
    delta: float

    def verifier(self) -> Actor:
        secret = secrets.token_bytes(32)
        digest = (yield from receive(Registration, "prover")).digest()
        salt = derive(secret, "verity/pouw/salt/v0", digest)
        shares = yield from receive(Shares, "prover", Send(Salt(salt), "prover"))
        valid = {n for n in shares.nonces if self._share(salt, digest, n)}
        return (Send(UsefulWork(digest, len(valid) << self.difficulty, self.delta)),)

    def prover(self, registration: Registration) -> Actor:
        digest, tries = (
            registration.digest(),
            sum(len(registration.circuits[e.circuit].gates) for e in registration.executions),
        )
        salt = (yield from receive(Salt, "verifier")).salt
        return (Send(Shares(tuple(n for n in range(tries) if self._share(salt, digest, n))), "verifier"),)

    def _share(self, salt: bytes, digest: str, nonce: int) -> bool:
        h = hashlib.sha256(salt + digest.encode() + nonce.to_bytes(8, "big")).digest()
        return int.from_bytes(h, "big") >> (256 - self.difficulty) == 0
