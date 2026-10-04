"""A toy commitment scheme (a hash of the leaves, opened by revealing them) and a toy challenge source (a hash of an
auditor secret). Stand-ins for `verity.commitments` and `verity.randomness`."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Mapping, Sequence

from verity.protocol.commitments import Commitment


class Sha256Leaves:
    name = "sha256-leaves"

    def commit(self, leaves: Sequence[bytes]) -> Commitment:
        return Commitment(self.name, hashlib.sha256(b"".join(hashlib.sha256(x).digest() for x in leaves)).digest())

    def open(self, leaves: Sequence[bytes], index: int) -> object:
        return list(leaves)

    def verify(self, commitment: Commitment, index: int, leaf: bytes, proof: object) -> bool:
        return proof[index] == leaf and self.commit(proof) == commitment


class AuditorSecret:
    name = "auditor-secret"

    def __init__(self, secret: bytes | None = None):
        self._secret = secret or secrets.token_bytes(32)

    def coins(self, domain: str, context: Mapping[str, str]) -> bytes:
        body = json.dumps([domain, sorted(context.items())]).encode()
        return hashlib.sha256(self._secret + body).digest()
