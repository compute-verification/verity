"""PoUS's session checked inside proofs (`pous.verifier.session_proofs`): the clear checks. An untrusted key, an unknown
scheme and a coin that doesn't open each reject."""

from __future__ import annotations

import dataclasses
import random

from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.resource_accounting.memory.pous.verifier import session_proofs as V

RNG = random.Random(3)
K, BLOCKS, BLOCK = 5, 64, 2056


def _rand(n: int) -> bytes:
    return bytes(RNG.getrandbits(8) for _ in range(n))


def _session(k: int = K):
    session = _rand(32)
    tree = mc.CoinTree.draw(b"source", session, BLOCKS, k, key=mc.coin_key(_rand(mc.HM96.key_bytes)),
                            salts=[_rand(mc.HM96.salt_bytes) for _ in range(k)])
    c = mc.SessionCommitment(session, tree.root, tree.key, BLOCKS, k, 854_000, BLOCK, _rand(64))
    return c, [tree.opening(j) for j in range(k)], _rand(16 + 64)


def _clear(c, openings, key, scheme=mc.LMS, trusted=None) -> tuple[str, ...]:
    return V.clear_reasons(c, openings, scheme, key, trusted={key} if trusted is None else trusted)


def test_the_clear_checks_reject() -> None:
    """A session's own key, scheme and openings pass; an untrusted key, an unknown scheme and a round whose coin
    doesn't open each reject."""
    c, openings, key = _session()
    assert _clear(c, openings, key) == ()
    assert "not a trusted" in " ".join(_clear(c, openings, key, trusted={bytes(80)}))
    assert "is not accepted" in " ".join(_clear(c, openings, key, scheme="test/x"))
    bad = openings[:2] + [dataclasses.replace(openings[2], index=(openings[2].index + 1) % BLOCKS)] + openings[3:]
    assert "round 2's index is not its committed coin" in _clear(c, bad, key)
