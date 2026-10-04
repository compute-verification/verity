"""Coins. A verifier draws from a secret only it holds, after the commitments the draw must follow are in its view, and
in a domain of its own, so no two draws share coins. Stands in for `verity.randomness.derive`."""

from __future__ import annotations

import hashlib


def derive(secret: bytes, domain: str, context: str) -> bytes:
    """32 bytes nobody without `secret` can predict, bound to `domain` and `context`."""
    return hashlib.sha256(b"\0".join((secret, domain.encode(), context.encode()))).digest()
