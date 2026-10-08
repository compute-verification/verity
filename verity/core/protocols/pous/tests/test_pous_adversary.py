"""Partial holdings replayed against real labels: measured rounds equal the schedule's, answers are bit-exact, and the
strategies of `docs/p3-cryptanalysis.md` behave as described."""

from __future__ import annotations

import hashlib
import random

import pytest

from verity.core.protocols.pous import adversary as A
from verity.core.protocols.pous.audit import challenge_key, simulate
from verity.core.protocols.pous.params import Params
from verity.core.protocols.pous.primitives import IdealTweakablePermutation, Primitives, RandomOracleKey, reference
from verity.core.protocols.pous.scheme import encode
from verity.core.protocols.pous.verifier import VerifierKey

PARAMS = Params(10, 64, 3)


def monolithic(params: Params) -> Primitives:
    """Lean's model: one `H` query per key."""
    mb = params.m // 8
    return Primitives(RandomOracleKey(lambda q: int.from_bytes(hashlib.shake_256(repr(q).encode()).digest(mb), "little")),
                      IdealTweakablePermutation(params.m, b"adversary-test"))


BACKENDS = {"sponge": reference, "monolithic": monolithic}


def _random_holding(rng: random.Random, n: int) -> A.Holding:
    return A.Holding(frozenset(w for w in range(n) if rng.random() < 0.15),
                     frozenset((w, u) for w in range(n) for u in range(n) if rng.random() < 0.3),
                     frozenset(u for u in range(n) if rng.random() < 0.6))


@pytest.mark.parametrize("backend", BACKENDS)
def test_random_holdings_replay(backend):
    prims = BACKENDS[backend](PARAMS)
    v = A.view(PARAMS, prims, random.Random(4).randbytes(PARAMS.w_bytes), b"\x05" * 24)
    rng = random.Random(9)
    for _ in range(12):
        m = A.measure("random", v, _random_holding(rng, PARAMS.n))
        assert m.correct


@pytest.mark.parametrize("backend", BACKENDS)
def test_honest_and_recompute(backend):
    prims = BACKENDS[backend](PARAMS)
    v = A.view(PARAMS, prims, bytes(PARAMS.w_bytes), b"\x06" * 24)
    honest = A.measure("honest", v, A.honest(PARAMS))
    assert honest.correct and set(honest.rounds) == {0} and honest.fraction == 1 and honest.game_rounds == 0
    empty = A.measure("recompute", v, A.recompute(PARAMS))
    assert empty.correct and empty.bits == 0 and min(empty.rounds) > 2 * PARAMS.n


def test_column_storage_meets_its_deadline():
    prims = reference(PARAMS)
    v = A.view(PARAMS, prims, bytes(PARAMS.w_bytes), b"\x07" * 24)
    for deadline in (24, 28, 40):
        hold = A.staggered_runs(PARAMS, prims.h, deadline, 2)
        m = A.measure("runs", v, hold)
        assert m.correct and max(m.rounds) <= deadline and m.saving > 0
    with pytest.raises(ValueError):
        A.column_storage(PARAMS, prims.h, range(PARAMS.n), 10)


def test_holding_state_is_a_responder():
    """A holding audited through the simulator: on time exactly when its answers fit the round budget."""
    prims = reference(PARAMS)
    w, salt = bytes(PARAMS.w_bytes), b"\x08" * 24
    enc = encode(PARAMS, prims, w, salt)
    vk = VerifierKey.build(enc)
    state = A.preprocess(A.view(PARAMS, prims, w, salt), A.staggered_runs(PARAMS, prims.h, 28, 2))
    key = challenge_key(b"audit", vk)
    worst = max(state.rounds(i) for i in range(PARAMS.blocks))
    assert simulate(vk, state, prims, key, k=60, rounds=worst).accepted
    assert not simulate(vk, state, prims, key, k=60, rounds=worst - 1).accepted


def test_lean_game_undercounts_reverse_and_promotion():
    """With a monolithic `H`, a meet-in-the-middle holding recovers every top within D = 20 oracle rounds, but Lean's
    column game needs 12 rounds, more than the ⌊(D + 1)/2⌋ = 10 N10′ grants: a reverse costs 1 oracle round and a
    promotion none, while the game charges a round for each."""
    params = Params(12, 64, 2)
    prims = monolithic(params)
    promoted = {w for w in range(12) if w >= 5 and (w - 5) % 7 < 2}
    hold = A.column_storage(params, prims.h, range(7, 12), 20, promoted=promoted)
    m = A.measure("mitm", A.view(params, prims, bytes(params.w_bytes), b"\x09" * 24), hold)
    assert m.correct and max(m.rounds) <= 20 and m.inverse > 0 and m.saving > 4
    assert not A.col_win(params, hold, 10) and m.game_rounds == 12
