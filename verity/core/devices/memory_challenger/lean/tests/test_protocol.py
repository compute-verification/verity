"""`verdict` as a core protocol (`MemoryChallenger.Protocol`): `memory-challenger-test serve` against `memory_challenger.verdict`
under `pous_test_signer.SCHEMES`, and `memory-challenger serve` against it under `memory_challenger.SCHEMES`, on honest
sessions, sessions under one adversarial edit (a late round, a wrong block digest, an index that isn't its opening's, a
tampered header, a bad signature, a wrong key, a replayed leaf, and more: `mc_cases.MUTATIONS`) and malformed cases."""

from __future__ import annotations

import dataclasses
import json
import shutil

import pytest
from hypothesis import strategies as st

import mc_cases as M
import verity_difftest as D
from verity.core.protocols.pous import memory_challenger as mc

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def signers_port(cases: st.SearchStrategy) -> D.ProtocolPort:
    return D.ProtocolPort("memory-challenger-test", M.protocol_reference(M.TEST_ID, M.ts.SCHEMES), cases, package=M.PACKAGE)


def deployed_port(cases: st.SearchStrategy) -> D.ProtocolPort:
    return D.ProtocolPort("memory-challenger", M.protocol_reference(mc.PROTOCOL, mc.SCHEMES), cases, package=M.PACKAGE)


signers_server = D.server_fixture(signers_port(st.none()))
deployed_server = D.server_fixture(deployed_port(st.none()))


@pytest.mark.slow
def test_the_test_protocol_agrees(signers_server):
    stats = D.run(signers_port(M.cases()), examples=400, server=signers_server)
    assert stats.counts.get("accept", 0) > 40 and stats.rejects > 200, stats


@pytest.mark.slow
def test_the_protocol_agrees_on_deployed_and_test_signed_sessions(deployed_server):
    """A session signed at S1's deployed parameters is accepted, and one a test signer signed never is."""
    cases = st.one_of(M.cases(M.deployed_signer()), M.cases(M.deployed_signer(), json_mutations=False), M.sessions().map(M.Session.case))
    stats = D.run(deployed_port(cases), examples=200, server=deployed_server)
    assert stats.counts.get("accept", 0) > 20 and stats.rejects > 100, stats


def test_the_named_rejections(signers_server):
    """Each of the task's adversarial sessions, once, by name: the verdict and its reasons as the Python gives them."""
    s = _honest()
    late = _honest(spends=[1, s.commitment.limit_ns + 1, 1])
    named = {
        "honest": s,
        "a late round": late,
        "a wrong block digest": M._round(s, 1, digest=lambda r: M._flip(r.digest, 3)),
        "an index that doesn't match its opening": M._round(s, 0, index=lambda r: (r.index + 1) % s.commitment.blocks),
        "a tampered header": M._unsigned(s, commitment=dataclasses.replace(s.commitment, limit_ns=s.commitment.limit_ns + 1)),
        "a bad signature": M._signature(s, M._flip(s.signed.signature, 100)),
        "a wrong key": M._stranger(s, 0),
        "a replayed leaf": M._replayed(s, 1),
    }
    port = signers_port(st.none())
    seen = {}
    for name, x in named.items():
        lean = D.replay(port, x.case(), signers_server)
        python = port.python(x.case())
        assert port.key(lean) == port.key(python), (name, lean, python)
        seen[name] = json.loads(lean[1])["reason"]
    assert seen["honest"] is None
    assert seen["a late round"] == f"the session ended late after 2 of 3 rounds; 1 answers are past Δ + RTT = {s.commitment.limit_ns} ns"
    assert seen["a wrong block digest"] == "1 answers are not their blocks of C"
    assert seen["an index that doesn't match its opening"].startswith("round 0's index is not its committed coin")
    assert seen["a tampered header"] == "the signature does not verify; the signed header is not the session's commitment"
    assert seen["a bad signature"] == seen["a replayed leaf"] == "the signature does not verify"
    assert seen["a wrong key"] == "the session is signed by a key that is not a trusted memory challenger's"


def _honest(spends=(1, 1, 1)) -> M.Session:
    """A three-round session over 8 blocks of 16 bytes, its limit 10 ns, signed at leaf 1 of a height-1 test key."""
    signer = M.test_signer(M.SECRETS[0], 1, 1)
    tree = mc.CoinTree.draw(b"source", bytes(32), 8, 3, key=M.COIN_KEYS[0], salts=[bytes([j]) * 192 for j in range(3)])
    clock = M.Clock(1_000)
    c = mc.MemoryChallenger(tree, signer, limit_ns=10, block_bytes=16, binding=b"vk", clock=clock)
    while not c.done:
        j = len(c.rounds)
        i = c.challenge()
        clock.now += spends[j]
        c.receive(M.block(i, 16))
    expected = tuple(mc.block_digest(M.block(i, 16)) for i in range(8))
    return M.Session(c.close(), c.commitment, c.openings(), expected, (signer.public_key,), signer)
