"""Proof of useful work: one salt from the verifier, drawn after the registration and before the stream, keys the work
of every execution the prover streams, however many there turn out to be and whatever their advice.

    verifier    takes the registration, sends a salt drawn from its secret, takes the stream, and bounds the work
    prover      once it has the salt, searches each execution's work for shares under that execution's key

Execution i's key is a hash of the salt, i and its root: fixed once the execution is, so nothing about the stream need
be known when the salt is drawn. What must be fixed before the salt is what the work is reused across: the builder and
the baseline it holds (in Pearl-C, the weights, whose salted forming is computed once per salt). The prover chooses
advice and inputs after the salt, so it can grind them for keys it likes; `verity_pouw`'s conjecture's error term holds
that grinding, and advice adds at most 2**advice_bits choices per execution to it.

Stands in for `verity_pouw`, whose shares are tiles of the streamed matmuls and whose key per matmul is derived the same
way inside the circuit (`ncp-v2`: a hash of the call's operand digest, the salt and the call's index), so a matmul the
advice adds is keyed like any other. Here a share is a nonce whose hash with its execution's key starts with
`difficulty` zero bits, one try per gate the execution ran, so the work bounded is hashing.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.primitives.events import Actor, Send, collect, receive
from verity.primitives.randomness import derive
from verity.primitives.registration import (
    Closed,
    Execution,
    Registration,
    built,
    digest,
)


@dataclass(frozen=True)
class Salt:
    salt: bytes


@dataclass(frozen=True)
class Shares:
    nonces: Mapping[int, tuple[int, ...]]  # per execution


@dataclass(frozen=True)
class UsefulWork:
    """About `work` tries went into the stream's keyed work, except with probability `delta`."""

    registration: str  # the stream's digest
    work: int
    delta: float


@dataclass(frozen=True)
class Pouw:
    difficulty: int  # leading zero bits a share's hash needs
    delta: float

    def verifier(self) -> Actor:
        secret = secrets.token_bytes(32)
        registration = yield from receive(Registration, "prover")
        salt = derive(secret, "verity/pouw/salt/v0", digest(registration))
        executions = yield from collect(Execution, Closed, "prover", Send(Salt(salt), "prover"))
        shares = (yield from receive(Shares, "prover")).nonces
        valid = sum(
            len({n for n in shares.get(i, ()) if self._share(_key(salt, i, e), n)}) for i, e in enumerate(executions)
        )
        return (Send(UsefulWork(digest(registration, executions), valid << self.difficulty, self.delta)),)

    def prover(self, registration: Registration, executions: Sequence[Execution]) -> Actor:
        salt = (yield from receive(Salt, "verifier")).salt
        nonces = {}
        for i, e in enumerate(executions):
            key, tries = _key(salt, i, e), len(built(registration.builder, e.advice)[0].gates)
            nonces[i] = tuple(n for n in range(tries) if self._share(key, n))
        return (Send(Shares(nonces), "verifier"),)

    def _share(self, key: bytes, nonce: int) -> bool:
        h = hashlib.sha256(key + nonce.to_bytes(8, "big")).digest()
        return int.from_bytes(h, "big") >> (256 - self.difficulty) == 0


def _key(salt: bytes, i: int, execution: Execution) -> bytes:
    return hashlib.sha256(b"verity/pouw/key/v0\0" + salt + i.to_bytes(8, "big") + execution.root.root).digest()
