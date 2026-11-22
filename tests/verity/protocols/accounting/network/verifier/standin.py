"""Fake upstream commitments for tests: hiding salts fixed by a seed, and a Commit's tree of rows under them."""

from __future__ import annotations

import hashlib
from typing import Sequence

from verity.commitments import hm96
from verity.protocols.accounting.network.verifier import io_tie as IT

_INST = hm96.SHA512


def fake_salt(seed: bytes, i: int) -> bytes:
    """Test data: a hiding salt fixed by `seed` and `i`, so that a fake upstream's tree is the same each run."""
    out = b""
    while len(out) < _INST.salt_bytes:
        out += hashlib.sha512(seed + i.to_bytes(8, "big") + len(out).to_bytes(4, "big")).digest()
    return out[:_INST.salt_bytes]


def fake_commit(rows: Sequence[bytes], seed: bytes = b"commit") -> tuple[tuple[bytes, ...], list[list[bytes]], bytes]:
    """A fake upstream Commit's tree of the rows under `fake_salt`s: (salts, levels, domain id)."""
    salts = tuple(fake_salt(seed, s) for s in range(len(rows)))
    dom = IT.domain(hashlib.sha512(seed).digest(), len(rows)).domain_id
    return salts, IT.rows_levels(dom, rows, salts), dom
