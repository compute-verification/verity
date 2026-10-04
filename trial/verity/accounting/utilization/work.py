"""Work: a lower bound on the compute the prover spent between a salt and a clock's reading. One salt, drawn after the
registration and before the stream, keys the work of every execution the prover streams, however many there turn out
to be and whatever their advice.

    verifier    takes the registration, sends a salt drawn from its secret, takes the stream, the shares, and the
                seconds `clock_from`'s clock read between the salt and the shares; records the bound
    prover      once it has the salt, searches each execution's work for shares under that execution's key
    clock       the reading, by whoever the verifier trusts for it: itself (remote) or a pinger at the site (onsite)

Useful work keys the prover's own declared computation; proof of work would key filler the verifier chose. Both bound
the same thing, compute busy after the salt and done when the clock was read; usefulness decides only whether the
compute that work covers was also the declared computation, which is integrity's to say.

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
import math
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.primitives.events import Actor, Send, collect, gather, receive, tell
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
class Elapsed:
    seconds: float


@dataclass(frozen=True)
class WorkBound:
    """At least `work` tries went into the stream's keyed work in the `seconds` between the salt and the shares, by
    `clock_from`'s clock; except with probability `delta`."""

    stream: str
    work: int
    seconds: float
    clock_from: str
    delta: float


@dataclass(frozen=True)
class Work:
    difficulty: int  # leading zero bits a share's hash needs
    delta: float

    def verifier(self, clock_from: str) -> Actor:
        secret = secrets.token_bytes(32)
        registration = yield from receive(Registration, "prover")
        salt = derive(secret, "verity/work/salt/v0", digest(registration))
        executions = yield from collect(Execution, Closed, "prover", Send(Salt(salt), "prover"))
        got = yield from gather({Shares: "prover", Elapsed: clock_from})
        shares, seconds = got[Shares].nonces, got[Elapsed].seconds
        valid = sum(
            len(
                {n for n in shares.get(i, ()) if 0 <= n < _budget(registration, e) and self._share(_key(salt, i, e), n)}
            )
            for i, e in enumerate(executions)
        )
        bound = WorkBound(digest(registration, executions), self._tries(valid), seconds, clock_from, self.delta)
        return (Send(bound),)

    def prover(self, registration: Registration, executions: Sequence[Execution]) -> Actor:
        salt = (yield from receive(Salt, "verifier")).salt
        nonces = {}
        for i, e in enumerate(executions):
            key = _key(salt, i, e)
            nonces[i] = tuple(n for n in range(_budget(registration, e)) if self._share(key, n))
        return (Send(Shares(nonces), "verifier"),)

    def clock(self, seconds: float) -> Actor:
        return tell(Send(Elapsed(seconds), "verifier"))

    def _share(self, key: bytes, nonce: int) -> bool:
        h = hashlib.sha256(key + nonce.to_bytes(8, "big")).digest()
        return int.from_bytes(h, "big") >> (256 - self.difficulty) == 0

    def _tries(self, shares: int) -> int:
        """The fewest tries that find `shares` shares with probability above `delta`. Each try is a share with
        probability p = 2**-difficulty, so by Bernstein's inequality fewer tries, with mean shares m, find
        m + t with probability at most exp(-t**2 / (2 * (m + t / 3))); this solves for that equal to `delta` at
        m + t = shares."""
        log = math.log(1 / self.delta)
        t = (math.sqrt(16 * log**2 / 9 + 8 * log * shares) - 4 * log / 3) / 2
        return max(0, int((shares - t) * 2**self.difficulty))


def _budget(registration: Registration, execution: Execution) -> int:
    """The tries an execution's work has: one per gate it ran, and none for advice that builds nothing."""
    try:
        return len(built(registration.builder, execution.advice)[0].gates)
    except ValueError:
        return 0


def _key(salt: bytes, i: int, execution: Execution) -> bytes:
    return hashlib.sha256(b"verity/work/key/v0\0" + salt + i.to_bytes(8, "big") + execution.root.root).digest()
