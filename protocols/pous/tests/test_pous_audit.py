"""Setup (commit, the verifier's salt, `vk` from its own encoding), the Merkle `vk`, and the timed audits."""

from __future__ import annotations

import hashlib
import random

import pytest

from verity_pous import setup
from verity_pous.audit import (DEADLINE_NS, HonestResponder, TimedVerifier, challenge_key, deadline_rounds, run_timed,
                               simulate)
from verity_pous.oracle import PQuery, Tweak, pure
from verity_pous.params import Params
from verity_pous.primitives import reference
from verity_pous.scheme import encode
from verity_pous.verifier import MerkleTree, VerifierKey, leaf_hash, node_hash, tree, verify_path

PARAMS = Params(8, 64, 3, 3)
PRIMS = reference(PARAMS)
W = random.Random(1).randbytes(PARAMS.w_bytes)


@pytest.fixture(scope="module")
def world():
    commitment = setup.commit(PARAMS, W)
    salt = setup.draw_salt(b"verifier-secret", commitment)
    enc = encode(PARAMS, PRIMS, W, salt)
    vk = setup.verifier_key(PARAMS, PRIMS, W, commitment, salt)
    return commitment, salt, enc, vk


def test_setup(world):
    commitment, salt, enc, vk = world
    assert len(salt) == 24 and salt == setup.draw_salt(b"verifier-secret", commitment)
    assert salt != setup.draw_salt(b"verifier-secret", setup.commit(PARAMS, bytes(PARAMS.w_bytes)))
    assert setup.check_server_root(vk, setup.server_root(enc))
    wrong = encode(PARAMS, PRIMS, W, bytes(24))
    assert not setup.check_server_root(vk, setup.server_root(wrong))
    with pytest.raises(ValueError):
        setup.verifier_key(PARAMS, PRIMS, bytes(PARAMS.w_bytes), commitment, salt)


def _rfc6962_root(leaves):
    if len(leaves) == 1:
        return leaves[0]
    k = 1 << ((len(leaves) - 1).bit_length() - 1)
    return node_hash(_rfc6962_root(leaves[:k]), _rfc6962_root(leaves[k:]))


@pytest.mark.parametrize("count", [1, 2, 3, 5, 8, 13, 24])
def test_merkle_paths(count):
    leaves = [hashlib.sha256(bytes([i])).digest() for i in range(count)]
    t = MerkleTree(leaves)
    assert t.root == _rfc6962_root(leaves)
    for i in range(count):
        assert verify_path(t.root, count, i, leaves[i], t.path(i))
        assert not verify_path(t.root, count, i, hashlib.sha256(b"x").digest(), t.path(i))
        if count > 1:
            assert not verify_path(t.root, count, (i + 1) % count, leaves[i], t.path(i))


def test_vk_is_the_spec(world):
    _, salt, enc, vk = world
    i = PARAMS.n + 5
    tag_s = salt + (1).to_bytes(8, "little")
    assert leaf_hash(PARAMS, salt, i, enc.block(i)) == hashlib.sha256(b"\x00" + tag_s + (5).to_bytes(2, "little")
                                                                     + enc.block(i)).digest()
    assert vk.root == tree(enc).root and len(vk.root) == 32
    assert vk.check(i, enc.block(i)) and not vk.check(i, enc.block(i + 1)) and not vk.check(i, b"")
    assert vk.check_path(i, enc.block(i), tree(enc).path(i))


def test_simulated_audit(world):
    _, _, enc, vk = world
    key = challenge_key(b"audit", vk)
    seen = []
    result = simulate(vk, HonestResponder(enc), PRIMS, key, k=40, on_answer=seen.append)
    assert result.accepted and set(result.rounds) == {0} and len(seen) == 40
    assert {a.index for a in result.answers} <= set(range(PARAMS.blocks))

    class Stalling:
        def answer(self, j, i):
            for _ in range(29):
                yield [PQuery(Tweak(0, 0, 0), 0)]
            return enc.blocks[i]

    late = simulate(vk, Stalling(), PRIMS, key, k=3, rounds=28)
    assert all(a.correct and not a.on_time for a in late.answers) and not late.accepted
    assert simulate(vk, Stalling(), PRIMS, key, k=3, rounds=29).accepted
    assert deadline_rounds() == 2 and deadline_rounds(500_000, 226_000) == 3
    assert not simulate(vk, Stalling(), PRIMS, key, k=1).accepted

    class Wrong:
        def answer(self, j, i):
            return pure(enc.blocks[(i + 1) % PARAMS.blocks])

    assert not simulate(vk, Wrong(), PRIMS, key, k=3).accepted


class FakeClock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


def test_timed_verifier(world):
    _, _, enc, vk = world
    key = challenge_key(b"audit", vk)
    responder = HonestResponder(enc, tree(enc))
    assert run_timed(TimedVerifier(vk, key, k=30), responder.block).accepted
    assert run_timed(TimedVerifier(vk, key, k=30, root_only=True), responder.block, responder.path).accepted

    assert DEADLINE_NS == 500_000
    clock = FakeClock()
    v = TimedVerifier(vk, key, k=3, rtt_allowance_ns=40_000, clock=clock)
    assert v.limit_ns == 540_000
    i = v.challenge()
    with pytest.raises(RuntimeError):
        v.challenge()
    clock.now += 540_000
    assert v.respond(enc.block(i)).on_time
    i = v.challenge()
    clock.now += 540_001
    late = v.respond(enc.block(i))
    assert late.correct and not late.on_time
    i = v.challenge()
    assert not v.respond(enc.block((i + 1) % PARAMS.blocks)).correct
    assert v.done and not v.accepted
    with pytest.raises(RuntimeError):
        v.challenge()


def test_challenges_are_bound(world):
    _, _, _, vk = world
    a = [challenge_key(b"audit", vk).uniform(PARAMS.blocks, j) for j in range(20)]
    b = [challenge_key(b"other", vk).uniform(PARAMS.blocks, j) for j in range(20)]
    assert a != b
    other = VerifierKey(vk.params, vk.salt, bytes(32), vk.leaves)
    assert a != [challenge_key(b"audit", other).uniform(PARAMS.blocks, j) for j in range(20)]
