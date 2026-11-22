"""PoUS's session checked inside proofs: the verifier's side. The memory attester's signed record never reaches it in
the clear: its message ``m`` (a hiding commitment to the record, salted) and the signature over it do. What it checks in
the clear is `clear_reasons`: the scheme, that the key is the one the configuration pins, the signature over ``m``
under that key, every round's coin opening, that the
store the session audits is the one the setup registered (its ``binding`` the store's registered root, its blocks the
store's rows), and that every round's index is one of the store's blocks, since the store's opening returns a verdict of
1 for a key past them (`gates.embedding`). `run.check_at` calls `memory_attester.verdict` until PoUS's verifier switches
to the session's proofs.
"""
from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from verity.protocols.accounting.memory.pous.attester import memory_attester as mc

__all__ = ["BAD_SIGNATURE", "OTHER_STORE", "PAST_STORE", "UNPINNED", "clear_reasons"]

#: a session whose signed binding or block count is not the registered store's
OTHER_STORE = "the session audits another store than the registered one: "
#: a round whose index is no block of the store
PAST_STORE = "an index past the store: "
#: a session signed under a key whose SHA-512 the configuration's pins don't give the memory attester
UNPINNED = "the session is signed by a key the configuration does not pin for the memory attester"
#: a signature that doesn't verify over the message under the pinned key
BAD_SIGNATURE = "the signature does not verify"

#: the LMS parameters (N, W, H) of each accepted scheme id, which the signature's unit is bound at
_LMS_PARAMS: Mapping[str, tuple[int, int, int]] = {mc.LMS: (64, 1, 20)}


def _verifies(verify: mc.Verify, public_key: bytes, session: bytes, message: bytes, signature: bytes) -> bool:
    try:
        return bool(verify(public_key, session, message, signature))
    except Exception:
        return False


def _by_j(openings: Sequence[mc.CoinOpening]) -> dict[int, mc.CoinOpening]:
    return {o.j: o for o in openings}


def clear_reasons(commitment: mc.SessionCommitment, openings: Sequence[mc.CoinOpening], scheme: str, public_key: bytes,
                  message: bytes, signature: bytes, *, store: Mapping[str, Any], pinned: Collection[str],
                  schemes: Mapping[str, mc.Verify] = mc.SCHEMES) -> tuple[str, ...]:
    """What the verifier checks in the clear: the scheme is accepted (and its program known), the key's SHA-512 is one
    of ``pinned``, the configuration's pins for the memory attester's PoUS sessions (`memory_attester.pinned`), never a
    key the session carries on its own (`UNPINNED`), and ``signature`` verifies over ``message`` under it
    (`BAD_SIGNATURE`); the session's signed ``binding`` and blocks are the root and rows of `store`, the store's registered
    entry as the verifier pinned it before any coin (`window.commit_store`'s ``entry()``), which its openings' table
    reads take (`OTHER_STORE`), and every round's opening, the last one for its j, opens its committed coin at an index
    below the store's blocks (`PAST_STORE`)."""
    reasons: list[str] = []
    if commitment.binding.hex() != store["root"]:
        reasons.append(f"{OTHER_STORE}its binding {commitment.binding.hex()[:16]}... is not the store's root "
                       f"{str(store['root'])[:16]}...")
    if commitment.blocks != store["leaves"]:
        reasons.append(f"{OTHER_STORE}it audits {commitment.blocks} blocks, the store has {store['leaves']}")
    if scheme not in schemes or scheme not in _LMS_PARAMS:
        reasons.append(f"signature scheme {scheme!r} is not accepted")
    elif hashlib.sha512(public_key).hexdigest() not in pinned:
        reasons.append(UNPINNED)
    elif not _verifies(schemes[scheme], public_key, commitment.session, message, signature):
        reasons.append(BAD_SIGNATURE)
    by_j = _by_j(openings)
    for j in range(commitment.k):
        o = by_j.get(j)
        if o is not None and not 0 <= o.index < store["leaves"]:
            reasons.append(f"{PAST_STORE}round {j}'s index {o.index} is not below the store's {store['leaves']} blocks")
        elif o is None or not mc.verify_coin(commitment, o):
            reasons.append(f"round {j}'s index is not its committed coin")
    return tuple(reasons)
