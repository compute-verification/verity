"""The memory attester's session coins: a commitment and its opening (the Lean is `Flock.SessionCoins`,
`verity/commitments/lean/Commitments/SessionCoins.lean`, which this mirrors).

A memory challenger commits to a session's coins before the session and opens them after it. Round ``j``'s coin is the index
of a block, hidden in an hm96 leaf under the session's key: ``x = SHA-512(COIN_TAG ‖ session ‖ u32 j ‖ u64 index)``, ``leaf =
HM96.tree_leaf(key, HM96.commit_string(key, x, salt))``, the leaves the positions of one frame-v3-sha512 tree under the domain
``SHA-512(COIN_BINDING ‖ session)[:32]`` (`domain`), whose root the session's commitment holds. `verify` is an opening's check,
`header_fields` and `header` the commitment's fields and the text the session's signature covers (how the session ended and its
number of rounds beside them), `signed_context` what the signature's message hashes before the record. The memory attester
(`memory_attester`) writes these and a supplied function's body (`registered.body`) reads them, and neither defines them.
Every tag is read from the domains' table (`verity.commitments.domains`).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from verity.commitments import domains as D
from verity.commitments import hm96
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.limits import VerificationLimits
from verity.commitments.merkle import Commitment, CommitmentDomain, Opening, verify_opening

__all__ = ["CLOCK", "COIN_BINDING", "COIN_SCHEMA", "COIN_TAG", "COIN_TREE", "COMPLETE", "HM96", "LATE", "PROTOCOL", "RECORDS",
           "coin_digest", "coin_leaf", "domain", "header", "header_fields", "signed_context", "verify"]

PROTOCOL = D.POUS_MEMORY_CHALLENGER
COIN_TREE = D.POUS_COIN_TREE_HM96_SHA512
RECORDS = "sha512/row/v2"
COIN_TAG = D.POUS_MEMORY_CHALLENGER_COIN.encode() + b"\0"
COIN_BINDING = D.POUS_MEMORY_CHALLENGER_COIN_TREE.encode() + b"\0"
COIN_SCHEMA = D.POUS_COIN_HM96_SHA512
COMPLETE, LATE = "complete", "late"
CLOCK = "CLOCK_MONOTONIC ns"
HM96 = hm96.SHA512
_LIMITS = VerificationLimits()


def domain(tag: bytes, session: bytes, count: int) -> CommitmentDomain:
    """The coin tree's domain: binding the first 32 bytes of ``SHA-512(tag ‖ session)``, owner -1, positions ``0 … count - 1``."""
    return CommitmentDomain(hashlib.sha512(tag + session).digest()[:32], -1, RangeIndexedDomain(0, count), hash="sha512")


def coin_digest(session: bytes, j: int, index: int) -> bytes:
    """``x_j``, round ``j``'s coin before hiding."""
    return hashlib.sha512(COIN_TAG + session + j.to_bytes(4, "big") + index.to_bytes(8, "big")).digest()


def coin_leaf(key: bytes, session: bytes, j: int, index: int, salt: bytes) -> bytes:
    """The hm96-sha512 leaf of round ``j``'s coin under the session's key ``key`` and the coin's salt."""
    return HM96.tree_leaf(key, HM96.commit_string(key, coin_digest(session, j, index), salt))


def verify(session: bytes, coin_root: bytes, coin_key: bytes, blocks: int, k: int, j: int, index: int, salt: bytes,
           path: Sequence[bytes]) -> bool:
    """The opening ``(j, index, salt, path)`` opens round ``j``'s coin under the root ``coin_root`` of ``k`` rounds over
    ``blocks`` blocks, to a block index; a round past ``k``, an index past ``blocks``, a key, salt or path of another form,
    or a path that doesn't climb to the root, opens nothing."""
    if not (0 <= j < k and 0 <= index < blocks):
        return False
    try:
        HM96.check_key(coin_key)
        if len(salt) != HM96.salt_bytes:
            return False
        leaf = coin_leaf(coin_key, session, j, index, salt)
        return verify_opening(domain(COIN_BINDING, session, k), Commitment(coin_root, k), Opening(j, leaf, tuple(path)),
                              COIN_SCHEMA, _LIMITS)
    except Exception:  # noqa: BLE001 - a malformed opening opens nothing
        return False


def header_fields(session: bytes, coin_root: bytes, coin_key_sha512: bytes, binding: bytes, blocks: int, k: int, limit_ns: int,
                  block_bytes: int) -> dict[str, Any]:
    """The session commitment's header: every field the signature covers but how the session ended and its rounds, the coin
    key by its hash, which the header holds from before the session (the key itself is released after it)."""
    return {"protocol": PROTOCOL, "coin_tree": COIN_TREE, "records": RECORDS, "session": session.hex(),
            "coin_root": coin_root.hex(), "coin_key_sha512": coin_key_sha512.hex(), "blocks": blocks, "k": k,
            "limit_ns": limit_ns, "block_bytes": block_bytes, "binding": binding.hex(), "clock": CLOCK}


def header(session: bytes, coin_root: bytes, coin_key_sha512: bytes, binding: bytes, blocks: int, k: int, limit_ns: int,
           block_bytes: int, ended: str, rounds: int) -> bytes:
    """The signed header: the commitment's header with how the session ended and its number of rounds, as
    ``json.dumps(sort_keys=True, separators=(",", ":"))`` writes it."""
    doc = {**header_fields(session, coin_root, coin_key_sha512, binding, blocks, k, limit_ns, block_bytes), "ended": ended,
           "rounds": rounds}
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


def signed_context(ident: bytes, session: bytes) -> bytes:
    """What ``lms.memory_attester_message`` hashes before the record: ``u64be(len tag) ‖ tag ‖ I ‖ u64be(len framed) ‖
    framed``, the tag the memory attester's LMS role and ``framed`` the session id framed (``u64be(len) ‖ session``)."""
    tag = D.MEMORY_CHALLENGER_RECORD.encode()
    framed = len(session).to_bytes(8, "big") + session
    return len(tag).to_bytes(8, "big") + tag + ident + len(framed).to_bytes(8, "big") + framed
