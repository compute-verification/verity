"""Proof of useful work: the verifier salts the registered work once integrity has accepted it, the prover returns the
shares its salted work turned up, and the verifier bounds the work from the valid ones.

    verifier    receives the registration integrity accepted, sends a salt drawn from its secret, and bounds the work
    prover      searches for shares under the salt

Stands in for `verity_pouw`, whose shares are tiles of the registered matmuls, checked by recomputation. Here a share is
a nonce whose hash with the salt and the registration starts with `difficulty` zero bits, one try per gate the
registered executions ran, so the work bounded is hashing.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Sequence
from dataclasses import dataclass

from verity.primitives.circuit import Circuit
from verity.primitives.events import Actor, Send, receive
from verity.primitives.randomness import derive
from verity.protocols.integrity.registration import Registered, Registration


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
    name: str = "pouw"

    def verifier(self, to: tuple[str, ...] = ()) -> dict[str, Actor]:
        return {f"verifier/{self.name}": self._verifier(to)}

    def prover(self, circuit: Circuit, inputs: Sequence[tuple[int, ...]], registration: Registration) -> dict[str, Actor]:
        return {f"prover/{self.name}": self._prover(registration.digest(), len(inputs) * len(circuit.gates))}

    def _verifier(self, to: tuple[str, ...]) -> Actor:
        secret = secrets.token_bytes(32)
        digest = (yield from receive(Registered, "verifier")).registration.digest()
        salt = derive(secret, "verity/pouw/salt/v0", digest)
        shares = yield from receive(Shares, "prover", Send(Salt(salt), to=(f"prover/{self.name}",)))
        valid = {n for n in shares.nonces if self._share(salt, digest, n)}
        return (Send(UsefulWork(digest, len(valid) << self.difficulty, self.delta), to=to),)

    def _prover(self, digest: str, tries: int) -> Actor:
        salt = (yield from receive(Salt, "verifier")).salt
        nonces = tuple(n for n in range(tries) if self._share(salt, digest, n))
        return (Send(Shares(nonces), to=(f"verifier/{self.name}",)),)

    def _share(self, salt: bytes, digest: str, nonce: int) -> bool:
        h = hashlib.sha256(salt + digest.encode() + nonce.to_bytes(8, "big")).digest()
        return int.from_bytes(h, "big") >> (256 - self.difficulty) == 0
