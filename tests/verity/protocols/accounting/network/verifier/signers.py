"""The attester's LMS signer for tests: the deployed scheme (N = 64, W = 1) on a tree of 2^6 leaves, so a key takes a
fraction of a second rather than the deployed H = 20's minutes; and the attester's commit-then-sign of a transfer
value's rows and of an execution phase's connection rows."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

from verity.commitments.crypto import lms
from verity.protocols.accounting.network.verifier import connection_rows as CR, transfer_rows as TR
from verity.protocols.accounting.network.verifier.certificate import Certificate, Direction, Issuer, LmsSigner
from verity.protocols.accounting.network.verifier.network_attester import NetworkAttesterParams, Released

SMALL = lms.Params(64, 1, 6)
#: the attester session the tests' issuers sign in, and another
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


def attest_transfer(issuer: Issuer, link: str, direction: Direction, window: int, kind: str, program: str, value: str,
                    rows: Sequence[Sequence[int]], salts: Sequence[bytes] | None = None, word_bits: int = TR.WORD_BITS,
                    schema: str = TR.SCHEMA) -> tuple[TR.Transfer, Certificate | None]:
    """The value's rows committed and their head signed by `issuer` for the link and execution phase (None when its
    signer refuses)."""
    t = TR.Transfer.commit(link, direction, window, kind, program, value, rows, salts, word_bits, schema)
    return t, issuer.sign(link, direction, window, t.head.digest())


def attest_connection_rows(issuer: Issuer, link: str, window: int, strides: Sequence[Sequence[Released]],
                           params: NetworkAttesterParams, salts: Mapping[str, Sequence[bytes]] | None = None
                           ) -> tuple[CR.Rows, Certificate | None]:
    """An execution phase's connection rows committed and their one head signed by `issuer` (None when its signer
    refuses)."""
    rows = CR.Rows.commit(link, window, CR.tables(params, strides), params, issuer.session, salts)
    return rows, issuer.sign(link, Direction.EGRESS, window, rows.head.digest())
