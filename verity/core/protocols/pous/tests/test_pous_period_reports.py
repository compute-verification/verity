"""`period_reports`: the memory challenger's sessions as sampling blocks of its party's certification period, under a small
HSS key (H = 4). Each session's one output is its liveness record; once the period closes, `verdict` accepts every
session against the closed period under the device's identity, and refuses an edited session, another period's root,
another deployment or party, and an identity it doesn't trust."""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

import pous_test_signer as ts
from test_pous_memory_challenger import BLOCK_BYTES, LIMIT, ROOT, Clock, coins, expected
from verity.core import attest
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period_reports as pr
from verity.ml.crypto import lms

SMALL = lms.Params(64, 1, 4)
DEPLOYMENT = hashlib.sha512(b"a deployment configuration").digest()
PARTY, LINK, ATTACHMENT = "memory-challenger", "server-3", bytes(range(32))


@pytest.fixture
def device(tmp_path) -> lms.Identity:
    return lms.Identity.init(tmp_path / "device", SMALL)


def _period(device: lms.Identity, sessions: int, period: int = 7):
    """`sessions` honest sessions signed by a `PeriodSigner` over one period, then the period closed."""
    signer = pr.PeriodSigner(attest.Period(device.issue(DEPLOYMENT, PARTY), ATTACHMENT, LINK, period), device.public)
    runs = []
    for s in range(sessions):
        clock = Clock()
        c = mc.MemoryChallenger(coins(hashlib.sha512(b"session %d %d" % (period, s)).digest()[:32]), signer,
                                log=ts.session_log(), limit_ns=LIMIT, block_bytes=BLOCK_BYTES, binding=ROOT, clock=clock)

        def answer(i: int) -> bytes:
            clock.now += 20_000
            return hashlib.shake_256(b"C%d" % i).digest(BLOCK_BYTES)

        runs.append((c, c.run(answer)))
    return runs, signer.close()


def _judge(c, signed, closed, *, trusted, deployment=DEPLOYMENT, party=PARTY):
    return mc.verdict(signed, c.commitment, c.openings(), expected, trusted=trusted,
                      schemes={pr.PERIOD: pr.period_verify(closed, deployment, party, SMALL)})


def test_each_session_is_a_block_whose_one_public_output_is_its_liveness_record(device) -> None:
    runs, closed = _period(device, 3)
    assert closed.records == tuple(signed.message() for _, signed in runs) and closed.root == attest.root(closed.records)
    for w, (_, signed) in enumerate(runs):
        assert signed.scheme == pr.PERIOD and signed.public_key == device.public
        live = lms.HssSignature.decode(SMALL, signed.signature)
        assert lms.verify_liveness(SMALL, device.public, DEPLOYMENT, PARTY, ATTACHMENT, LINK, 7, w, live)
        assert not lms.verify_liveness(SMALL, device.public, DEPLOYMENT, PARTY, ATTACHMENT, LINK, 7, w + 1, live)


def test_every_session_is_accepted_against_the_closed_period(device) -> None:
    runs, closed = _period(device, 3)
    for c, signed in runs:
        v = _judge(c, signed, closed, trusted={device.public})
        assert v.accepted, v.reasons


def test_what_the_period_did_not_sign_is_refused(device) -> None:
    runs, closed = _period(device, 2)
    _, other = _period(device, 2, period=8)
    c, signed = runs[0]
    cases = {"edited": _judge(c, dataclasses.replace(signed, ended=mc.LATE), closed, trusted={device.public}),
             "another block's liveness record": _judge(c, dataclasses.replace(signed, signature=runs[1][1].signature),
                                                       closed, trusted={device.public}),
             "another period": _judge(c, signed, other, trusted={device.public}),
             "another deployment": _judge(c, signed, closed, trusted={device.public},
                                          deployment=hashlib.sha512(b"another").digest()),
             "another party": _judge(c, signed, closed, trusted={device.public}, party="network-certifier")}
    for name, v in cases.items():
        assert "the signature does not verify" in v.reasons, name
    untrusted = _judge(c, signed, closed, trusted={bytes(80)})
    assert not untrusted.accepted and any("not a trusted" in r for r in untrusted.reasons)


def test_the_period_scheme_is_not_accepted_by_default(device) -> None:
    runs, _ = _period(device, 1)
    c, signed = runs[0]
    v = mc.verdict(signed, c.commitment, c.openings(), expected, trusted={device.public})
    assert not v.accepted and any("not accepted" in r for r in v.reasons)


def test_a_challenger_restarted_within_its_attachment_signs_no_more_sessions(device) -> None:
    """Under no-state-rollback: the challenger's party key reopened from the device within the period, as a restarted
    challenger would, is refused the period's attachment, so it opens no period of it and signs no session of it; the
    party it would replace signs on."""
    signer = pr.PeriodSigner(attest.Period(device.issue(DEPLOYMENT, PARTY), ATTACHMENT, LINK, 7), device.public)
    signer.sign(b"session 0", bytes(64))
    restarted = device.party(DEPLOYMENT, PARTY)
    with pytest.raises(lms.StateError, match="already started"):
        pr.PeriodSigner(attest.Period(restarted, ATTACHMENT, LINK, 7), device.public)
    with pytest.raises(lms.StateError, match="not the one this key started"):
        restarted.liveness(ATTACHMENT, LINK, 7, 1)
    signer.sign(b"session 1", bytes(range(64)))
    assert signer.close().count == 2
