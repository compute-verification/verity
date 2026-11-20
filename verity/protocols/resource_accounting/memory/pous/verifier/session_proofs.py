"""PoUS's session checked inside proofs: the verifier's side. The memory attester's signed record never reaches it in
the clear. What it checks in the clear is `clear_reasons`: the scheme, the key, every round's coin opening, that the
store the session audits is the one the setup registered (its ``binding`` the store's registered root, its blocks the
store's rows), and that every round's index is one of the store's blocks, since the store's opening returns a verdict of
1 for a key past them (`gates.embedding`). `run.check_at` calls `memory_attester.verdict` until PoUS's verifier switches
to the session's proofs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc

__all__ = ["OTHER_STORE", "PAST_STORE", "clear_reasons"]

#: a session whose signed binding or block count is not the registered store's
OTHER_STORE = "the session audits another store than the registered one: "
#: a round whose index is no block of the store
PAST_STORE = "an index past the store: "

#: the LMS parameters (N, W, H) of each accepted scheme id, which the signature's unit is bound at
_LMS_PARAMS: Mapping[str, tuple[int, int, int]] = {mc.LMS: (64, 1, 20)}


def _by_j(openings: Sequence[mc.CoinOpening]) -> dict[int, mc.CoinOpening]:
    return {o.j: o for o in openings}


def clear_reasons(commitment: mc.SessionCommitment, openings: Sequence[mc.CoinOpening], scheme: str, public_key: bytes,
                  *, store: Mapping[str, Any], trusted: frozenset[bytes] | set[bytes],
                  schemes: Mapping[str, mc.Verify] = mc.SCHEMES) -> tuple[str, ...]:
    """What the verifier checks in the clear: the scheme is accepted (and its program known), the key is a trusted memory
    attester's, the session's signed ``binding`` and blocks are the root and rows of `store`, the store's registered
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
    elif public_key not in trusted:
        reasons.append("the session is signed by a key that is not a trusted memory challenger's")
    by_j = _by_j(openings)
    for j in range(commitment.k):
        o = by_j.get(j)
        if o is not None and not 0 <= o.index < store["leaves"]:
            reasons.append(f"{PAST_STORE}round {j}'s index {o.index} is not below the store's {store['leaves']} blocks")
        elif o is None or not mc.verify_coin(commitment, o):
            reasons.append(f"round {j}'s index is not its committed coin")
    return tuple(reasons)
