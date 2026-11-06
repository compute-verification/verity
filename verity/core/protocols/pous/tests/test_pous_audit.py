"""Setup (commit, the verifier's salt, `vk` from its own encoding), the Merkle `vk`, and the timed audits."""

from __future__ import annotations

import hashlib
import itertools
import random

import pytest

from pous_fakes import Clock
from verity.ml.commitments.rowleaf import sha512_row_v2_digest, sha512_row_v2_prefix
from verity.core.protocols.pous import setup
from verity.core.protocols.pous.audit import (DEADLINE_NS, HonestResponder, UnsignedRounds, challenge_key, deadline_rounds, run_timed,
                               simulate)
from verity.core.protocols.pous.oracle import PQuery, Tweak, pure
from verity.core.protocols.pous.params import Params
from verity.core.protocols.pous.primitives import reference
from verity.core.protocols.pous.scheme import encode
from verity.core.protocols.pous.verifier import (MerkleTree, VerifierKey, leaf_hash, node_hash, path_length, tree,
                                                 verify_path)

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
    leaves = [hashlib.sha512(bytes([i])).digest() for i in range(count)]
    t = MerkleTree(leaves)
    assert t.root == _rfc6962_root(leaves)
    for i in range(count):
        assert verify_path(t.root, count, i, leaves[i], t.path(i))
        assert path_length(count, i) == len(t.path(i))
        assert not verify_path(t.root, count, i, hashlib.sha512(b"x").digest(), t.path(i))
        if count > 1:
            assert not verify_path(t.root, count, (i + 1) % count, leaves[i], t.path(i))
    for i in (-1, count):
        with pytest.raises(ValueError):
            path_length(count, i)


def test_a_path_with_a_sibling_that_is_not_a_digest_is_refused():
    """Every sibling is a 64-byte node digest (the P2 spec's RESPONSE, `sibling[64]`): a path whose sibling has another
    length is refused even when the root was built from it."""
    leaf = hashlib.sha512(b"leaf").digest()
    for sibling in (b"x" * 63, b"x" * 65, b""):
        assert not verify_path(node_hash(leaf, sibling), 2, 0, leaf, [sibling])
        assert not verify_path(node_hash(sibling, leaf), 2, 1, leaf, [sibling])


def test_vk_is_the_spec(world):
    _, salt, enc, vk = world
    i = PARAMS.n + 5
    tag_s = salt + (1).to_bytes(8, "little")
    row = hashlib.sha512(sha512_row_v2_prefix(8 * PARAMS.block_bytes) + enc.block(i)).digest()
    assert row == sha512_row_v2_digest(enc.block(i), 8 * PARAMS.block_bytes)
    assert leaf_hash(PARAMS, salt, i, enc.block(i)) == hashlib.sha512(b"\x00" + tag_s + (5).to_bytes(2, "little")
                                                                     + row).digest()
    assert vk.root == tree(enc).root and len(vk.root) == 64
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


def test_memory_challenger(world):
    _, _, enc, vk = world
    key = challenge_key(b"audit", vk)
    responder = HonestResponder(enc, tree(enc))
    assert run_timed(UnsignedRounds(vk, key, k=30, clock=itertools.count(0, 1000).__next__), responder.block).accepted
    assert run_timed(UnsignedRounds(vk, key, k=30, root_only=True, clock=itertools.count(0, 1000).__next__), responder.block, responder.path).accepted

    assert DEADLINE_NS == 500_000
    clock = Clock()
    v = UnsignedRounds(vk, key, k=3, rtt_allowance_ns=40_000, clock=clock)
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
