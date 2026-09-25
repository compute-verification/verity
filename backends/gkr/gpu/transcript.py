"""SHA-256 Fiat-Shamir transcript, the structure of ``backends/gkr/src/transcript.rs`` over BabyBear^6.

``state <- H(state || message)``; ``challenge_i <- H(state || "chal" || ctr)`` read as eight little-endian
32-bit words, each accepted iff below ``p`` (rejection sampling), six accepted words per extension challenge.
An extension element is absorbed as 24 bytes (six LE u32 coordinates).  ``slots`` counts every challenge
drawn (``rounds.protocol``).  Fiat-Shamir is what the CPU prover uses too (the interactive object and step 0 of
PROTOCOL.md are not implemented in either); the diagnostic label stays ``NON_ZK_PROOF_DIAGNOSTIC``.
"""
from __future__ import annotations

import hashlib

from .field import DEG, Ext, P, exts_to_bytes, to_bytes

DOMAIN = b"verity-gkr/candidate-a/v1/babybear6"


class Transcript:
    def __init__(self, domain: bytes = DOMAIN) -> None:
        self.state = hashlib.sha256(domain).digest()
        self.ctr = 0
        self.slots = 0

    def absorb(self, data: bytes) -> None:
        self.state = hashlib.sha256(self.state + data).digest()

    def absorb_ext(self, x: Ext) -> None:
        self.absorb(to_bytes(x))

    def absorb_exts(self, xs) -> None:
        self.absorb(exts_to_bytes(xs))

    def _words(self):
        while True:
            d = hashlib.sha256(self.state + b"chal" + self.ctr.to_bytes(8, "little")).digest()
            self.ctr += 1
            for i in range(8):
                v = int.from_bytes(d[4 * i:4 * i + 4], "little")
                if v < P:
                    yield v

    def challenge(self) -> Ext:
        self.slots += 1
        g = self._words()
        return tuple(next(g) for _ in range(DEG))

    def challenges(self, n: int) -> list[Ext]:
        return [self.challenge() for _ in range(n)]
