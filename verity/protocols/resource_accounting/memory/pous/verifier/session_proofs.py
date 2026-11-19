"""PoUS's session checked inside proofs: the verifier's side. The memory attester's signed record never reaches it in
the clear. What it checks in the clear is `clear_reasons`: the scheme, the key and every round's coin opening.
`run.check_at` calls `memory_attester.verdict` until PoUS's verifier switches to the session's proofs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc

__all__ = ["clear_reasons"]

#: the LMS parameters (N, W, H) of each accepted scheme id, which the signature's unit is bound at
_LMS_PARAMS: Mapping[str, tuple[int, int, int]] = {mc.LMS: (64, 1, 20)}


def _by_j(openings: Sequence[mc.CoinOpening]) -> dict[int, mc.CoinOpening]:
    return {o.j: o for o in openings}


def clear_reasons(commitment: mc.SessionCommitment, openings: Sequence[mc.CoinOpening], scheme: str, public_key: bytes,
                  *, trusted: frozenset[bytes] | set[bytes], schemes: Mapping[str, mc.Verify] = mc.SCHEMES) -> tuple[str, ...]:
    """What the verifier checks in the clear: the scheme is accepted (and its program known), the key is a trusted memory
    attester's, and every round's opening, the last one for its j, opens its committed coin."""
    reasons: list[str] = []
    if scheme not in schemes or scheme not in _LMS_PARAMS:
        reasons.append(f"signature scheme {scheme!r} is not accepted")
    elif public_key not in trusted:
        reasons.append("the session is signed by a key that is not a trusted memory challenger's")
    by_j = _by_j(openings)
    for j in range(commitment.k):
        o = by_j.get(j)
        if o is None or not mc.verify_coin(commitment, o):
            reasons.append(f"round {j}'s index is not its committed coin")
    return tuple(reasons)
