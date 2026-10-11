"""The certifier's LMS signer for tests: the deployed scheme (N = 64, W = 1) on a tree of 2^6 leaves, so a key takes a
fraction of a second rather than the deployed H = 20's minutes."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from verity.ml.crypto import lms
from verity.core.protocols.certifier.certificate import LmsSigner

SMALL = lms.Params(64, 1, 6)
#: the certifier session the tests' issuers sign in, and another
SESSION, OTHER_SESSION = bytes(range(32)), bytes(range(1, 33))


@lru_cache(maxsize=None)
def key(seed: bytes = b"certifier") -> lms.PrivateKey:
    return lms.keygen(SMALL, hashlib.sha512(seed).digest())


def signer(directory: Path, seed: bytes = b"certifier") -> LmsSigner:
    """A signer with fresh leaf state (next leaf 0) under `directory`."""
    return LmsSigner.create(key(seed), directory / "signer-state.json")


def used(s: LmsSigner) -> list[int]:
    """The leaves `s`'s state has journalled as used."""
    return [int(q) for q in s.state.with_name(s.state.name + ".journal").read_text().split()]
