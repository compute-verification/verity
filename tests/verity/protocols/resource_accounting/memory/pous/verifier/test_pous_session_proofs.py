"""PoUS's session checked inside proofs (`pous.verifier.session_proofs`): the clear checks. An untrusted key, an unknown
scheme, a coin that doesn't open, a session over another store than the registered one and an index past the store each
reject."""

from __future__ import annotations

import dataclasses
import random

from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.resource_accounting.memory.pous.verifier import session_proofs as V

RNG = random.Random(3)
K, BLOCKS, BLOCK = 5, 64, 2056


def _rand(n: int) -> bytes:
    return bytes(RNG.getrandbits(8) for _ in range(n))


STORE = {"value": "store", "root": _rand(64).hex(), "leaves": BLOCKS}


def _session(k: int = K, drawn: int = BLOCKS):
    """A session of `k` rounds over the registered `STORE`, its coins drawn over `drawn` blocks."""
    session = _rand(32)
    tree = mc.CoinTree.draw(b"source", session, drawn, k, key=mc.coin_key(_rand(mc.HM96.key_bytes)),
                            salts=[_rand(mc.HM96.salt_bytes) for _ in range(k)])
    c = mc.SessionCommitment(session, tree.root, tree.key, BLOCKS, k, 854_000, BLOCK, bytes.fromhex(STORE["root"]))
    return c, [tree.opening(j) for j in range(k)], _rand(16 + 64)


def _clear(c, openings, key, scheme=mc.LMS, trusted=None, store=STORE) -> tuple[str, ...]:
    return V.clear_reasons(c, openings, scheme, key, store=store, trusted={key} if trusted is None else trusted)


def test_the_clear_checks_reject() -> None:
    """A session's own key, scheme and openings pass; an untrusted key, an unknown scheme and a round whose coin
    doesn't open each reject."""
    c, openings, key = _session()
    assert _clear(c, openings, key) == ()
    assert "not a trusted" in " ".join(_clear(c, openings, key, trusted={bytes(80)}))
    assert "is not accepted" in " ".join(_clear(c, openings, key, scheme="test/x"))
    bad = openings[:2] + [dataclasses.replace(openings[2], index=(openings[2].index + 1) % BLOCKS)] + openings[3:]
    assert "round 2's index is not its committed coin" in _clear(c, bad, key)


def test_a_session_over_another_store_is_refused() -> None:
    """The store the session's openings pin is the registered one, whose root the setup published: a session whose
    signed binding is another root, or whose block count is not the store's rows, is refused by name."""
    c, openings, key = _session()
    other = {**STORE, "root": _rand(64).hex()}
    (why,) = _clear(c, openings, key, store=other)
    assert why.startswith(V.OTHER_STORE) and "is not the store's root" in why
    (why,) = _clear(c, openings, key, store={**STORE, "leaves": BLOCKS + 1})
    assert why.startswith(V.OTHER_STORE) and f"it audits {BLOCKS} blocks, the store has {BLOCKS + 1}" in why


def test_an_index_past_the_store_is_refused() -> None:
    """The store's opening returns a verdict of 1 for a key past its rows, so the index bound is a clear check: coins
    that open, drawn over twice the store's blocks, are refused by name at each round past it."""
    c, openings, key = _session(k=16, drawn=2 * BLOCKS)
    past = [o.j for o in openings if o.index >= BLOCKS]
    assert past and _clear(c, openings, key) == tuple(
        f"{V.PAST_STORE}round {j}'s index {openings[j].index} is not below the store's {BLOCKS} blocks" for j in past)
