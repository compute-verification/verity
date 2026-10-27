"""The network certifier on device-attest (`period_root`): every digest `Issuer` would have certified is a record of its
certification period, under one signed root, and each link-window's one public output is its liveness record. The toy HSS
keys are the deployed scheme (N = 64, W = 1) on trees of 2^4 leaves."""

import dataclasses
import hashlib
import itertools
import json
from fractions import Fraction

import pytest

from verity.core import attest
from verity.core.protocols.certifier.certificate import (SALT_BYTES, Direction, Issuer, SignerRefused,
                                                         certificate_digest, certificate_message, clock_commitment)
from verity.core.protocols.certifier.grid import COMPLETE, GridParams, Record, fill
from verity.core.protocols.certifier.period_root import PARTY, Entry, PeriodIssuer, refusal
from verity.core.protocols.certifier.schedule import ClockSync
from verity.ml.crypto import lms

SMALL = lms.Params(64, 1, 4)
DEPLOYMENT, OTHER_DEPLOYMENT = (hashlib.sha512(b"a deployment configuration").digest(),
                                hashlib.sha512(b"another deployment configuration").digest())
ATTACHMENT, OTHER_ATTACHMENT = bytes(range(32)), bytes(range(1, 33))
LINK, M = "gpu0-gpu1", 3
P = GridParams(buckets=3, allocation=2, frame_bytes=2, bucket_ms=50)
PARAMS = bytes(range(64))
GRID = fill([b"ab", b"cd", b"ef"], [2, 0, 1], 2)
SYNC = ClockSync(1, Fraction(1, 2))
EGRESS, INGRESS = Direction.EGRESS, Direction.INGRESS


def coins():
    counter = itertools.count()
    return lambda n: hashlib.sha512(b"coin %d" % next(counter)).digest()[:n]


@pytest.fixture
def device(tmp_path):
    return lms.Identity.init(tmp_path / "device", SMALL)


@pytest.fixture
def key(device):
    return device.issue(DEPLOYMENT, PARTY)


def certify(issuer, w):
    """A rows-form link-window's digests: the responses record (egress), its clock, and a cells head's digest."""
    rec = issuer.certify(Record("responses", w, GRID, COMPLETE), EGRESS, P, PARAMS)
    clock = issuer.salted_clock("responses", w, SYNC)
    cells = issuer.sign("requests", INGRESS, w, hashlib.sha512(b"cells head %d" % w).digest())
    return rec, clock, cells


def run(key, periods=2, first=0):
    issuer = PeriodIssuer(key, ATTACHMENT, LINK, M, first, coins=coins())
    lives = []
    for w in range(first, first + periods * M):
        certify(issuer, w)
        lives.append(issuer.live(w))
    return issuer, lives


def check(device, closed, lives, entries=None, **kw):
    args = dict(identity=device.public, deployment=DEPLOYMENT, attachment=ATTACHMENT, link=LINK,
                period=closed.period, windows=M, params=SMALL) | kw
    liveness = {l.block: l.signature for l in lives if l is not None and l.period == closed.period}
    if entries is None:
        entries = [e for l in lives if l is not None and l.period == closed.period for e in l.entries]
    return refusal(closed, liveness, entries, **args)


def test_each_digest_is_its_certificates_message_and_a_record_of_its_period(device, key):
    """Records are the messages `Issuer`'s certificates would have signed (same coins, same key id), in order, and every
    period closes after its last block with a root its records climb to."""
    issuer, lives = run(key)
    assert [p.period for p in issuer.close()] == [0, 1] and issuer.refused == []
    plain = Issuer(_Message(key.key.I), ATTACHMENT, coins=coins())
    for w in range(2 * M):
        certify(plain, w)
    assert [e.record for l in lives for e in l.entries] == plain.signer.messages
    for p, closed in enumerate(issuer.closed):
        assert closed.records == tuple(e.record for l in lives[p * M:(p + 1) * M] for e in l.entries)
        assert closed.count == 3 * M and closed.root == attest.root(closed.records)
        for i in range(closed.count):
            r = closed.read(i)
            assert attest.verify_opening(closed.root, closed.count, i, r.record, r.path)
        assert check(device, closed, lives) is None
    assert [(l.window, l.period, l.block) for l in lives] == [(w, w // M, w % M) for w in range(2 * M)]


class _Message:
    """A signer that keeps the messages it is asked to sign."""

    def __init__(self, key_id):
        self.key_id, self.messages = key_id, []

    def start(self, session):
        pass

    def sign(self, message):
        self.messages.append(message)
        return b""


def test_the_entries_open_as_the_certificates_did(key):
    """The record head's salted digest and the clock's salted digest are the certificates' digests, and each record is
    `certificate_message` of its entry's fields under the lower key's identifier in the attachment."""
    issuer = PeriodIssuer(key, ATTACHMENT, LINK, M, coins=coins())
    rec, clock, cells = certify(issuer, 0)
    assert len(rec.salt) == len(clock.salt) == SALT_BYTES
    assert rec.entry.digest == certificate_digest(rec.salt, rec.head.digest())
    assert clock.entry.digest == clock_commitment(clock.salt, clock.sync)
    for e, link, d in ((rec.entry, "responses", EGRESS), (clock.entry, "responses", EGRESS),
                       (cells, "requests", INGRESS)):
        assert (e.link, e.direction, e.window) == (link, d, 0)
        assert e.record == certificate_message(key.key.I, ATTACHMENT, e.link, e.direction, e.window, e.digest)
    assert issuer.live(0).entries == (rec.entry, clock.entry, cells)


def test_each_liveness_record_verifies_in_the_clear_and_names_only_its_block(device, key):
    _, lives = run(key)
    for l in lives:
        sig = lms.HssSignature.decode(SMALL, l.signature)
        assert lms.verify_liveness(SMALL, device.public, DEPLOYMENT, PARTY, ATTACHMENT, LINK, l.period, l.block, sig)
        assert not lms.verify_liveness(SMALL, device.public, DEPLOYMENT, PARTY, ATTACHMENT, LINK, l.period,
                                       (l.block + 1) % M, sig)


def test_a_period_whose_root_or_signature_is_not_its_own_is_refused(tmp_path, device, key):
    issuer, lives = run(key)
    closed = issuer.closed[0]
    assert check(device, closed, lives, period=1).startswith("another period's")
    assert check(device, closed, lives, attachment=OTHER_ATTACHMENT).startswith("another period's")
    assert check(device, dataclasses.replace(closed, records=closed.records[:-1]), lives) == "its root isn't its records'"
    unsigned = "its root's signature doesn't verify under the device's identity"
    assert check(device, closed, lives, deployment=OTHER_DEPLOYMENT) == unsigned
    other = lms.Identity.init(tmp_path / "another device", SMALL)
    assert check(other, closed, lives) == unsigned
    truncated = dataclasses.replace(closed, signature=closed.signature[:-1])
    assert check(device, truncated, lives).startswith(f"{unsigned}: ")
    assert check(device, dataclasses.replace(closed, signature=issuer.closed[1].signature), lives) == unsigned


def test_a_liveness_record_that_is_not_its_blocks_is_refused(device, key):
    issuer, lives = run(key)
    closed = issuer.closed[0]
    swapped = [dataclasses.replace(lives[0], signature=lives[1].signature)] + lives[1:]
    assert check(device, closed, swapped) == "block 0's liveness record doesn't verify under the device's identity"
    later = [dataclasses.replace(lives[0], signature=lives[M].signature)] + lives[1:]
    assert check(device, closed, later) == "block 0's liveness record doesn't verify under the device's identity"
    extra = lives + [dataclasses.replace(lives[0], block=M)]
    assert check(device, closed, extra) == f"a liveness record for block {M}, which a period of {M} hasn't"


def test_an_entry_that_is_not_a_certified_record_of_the_period_is_refused(device, key):
    issuer, lives = run(key)
    closed, e = issuer.closed[0], lives[1].entries[0]
    what = f"{e.link} egress link-window 1"
    edited = dataclasses.replace(e, digest=bytes(64))
    assert check(device, closed, lives, [edited]) == f"{what}: its record isn't its fields' under the root's party key"
    moved = dataclasses.replace(e, window=2)
    assert check(device, closed, lives, [moved]).endswith("its record isn't its fields' under the root's party key")
    forged = dataclasses.replace(edited, record=certificate_message(key.key.I, ATTACHMENT, e.link, e.direction, 1,
                                                                    bytes(64)))
    assert check(device, closed, lives, [forged]) == f"{what}: not a record of the period"
    later = lives[M].entries[0]
    assert check(device, closed, lives, [later]) == f"{later.link} egress link-window {M}: not a link-window of the period"
    assert check(device, closed, lives[:1] + lives[2:], [e]) == f"{what}: its block, 1, has no liveness record"
    assert check(device, closed, lives[:1] + lives[2:], list(lives[0].entries) + list(lives[2].entries)) is None


def test_a_link_window_never_live_is_certified_nowhere(device, key):
    """A link-window the certifier closed no block for (a refused connection, a run's end) puts nothing in any period:
    its entries are dropped, and its block has no liveness record."""
    issuer = PeriodIssuer(key, ATTACHMENT, LINK, M, coins=coins())
    certify(issuer, 0)
    _, _, cells = certify(issuer, 1)
    live = issuer.live(1)
    assert (live.block, live.entries[-1]) == (1, cells)
    with pytest.raises(ValueError, match="not after link-window 1"):
        certify(issuer, 1)
    certify(issuer, 2)
    [closed] = issuer.close()
    assert closed.records == tuple(e.record for e in live.entries) and issuer.closed == [closed]
    assert check(device, closed, [live]) is None
    assert issuer.close() == [closed]


def test_a_period_starts_where_its_attachment_does_and_closes_early_on_a_later_period(device, key):
    issuer = PeriodIssuer(key, ATTACHMENT, LINK, M, first=10, coins=coins())
    with pytest.raises(ValueError, match="before the attachment's first, 10"):
        certify(issuer, 9)
    certify(issuer, 11)
    first = issuer.live(11)
    assert (first.period, first.block) == (0, 1)
    certify(issuer, 10 + M)
    later = issuer.live(10 + M)
    assert (later.period, later.block) == (1, 0)
    [closed] = issuer.closed
    assert closed.period == 0 and closed.records == tuple(e.record for e in first.entries)
    assert check(device, closed, [first, later], first=10) is None


def test_an_attachment_is_started_once_on_the_party_key(device, key):
    """A second issuer of the attachment, such as a restarted certifier, is refused before it signs, so no period of the
    attachment gets two roots; another attachment starts."""
    PeriodIssuer(key, ATTACHMENT, LINK, M)
    with pytest.raises(SignerRefused, match="already started"):
        PeriodIssuer(device.party(DEPLOYMENT, PARTY), ATTACHMENT, LINK, M)
    issuer = PeriodIssuer(key, OTHER_ATTACHMENT, LINK, M, coins=coins())
    certify(issuer, 0)
    assert issuer.live(0) is not None


def test_an_issuer_on_another_link_shares_the_attachments_start_and_keeps_periods_of_its_own(device, key):
    """`PeriodIssuer.on`: the warden's load on its uplink beside the warden's link in one attachment, which a second
    issuer would be refused (above). Its records and liveness records are its link's period's, and no other link's; an
    entry rebuilds from its public fields (`Entry.of`), and entries and liveness records read back from their JSON."""
    issuer = PeriodIssuer(key, ATTACHMENT, LINK, M, coins=coins())
    uplink = issuer.on("s0/uplink")
    head = uplink.sign("s0/uplink", INGRESS, 0, hashlib.sha512(b"a load head").digest())
    live = uplink.live(0)
    [closed] = uplink.close()
    assert (closed.link, closed.period, closed.records) == ("s0/uplink", 0, (head.record,))
    assert check(device, closed, [live], link="s0/uplink") is None
    assert check(device, closed, [live]).startswith("another period's")
    certify(issuer, 0)
    assert issuer.live(0).block == 0 and issuer.closed == [] and uplink.closed == [closed]
    assert Entry.of(key.key.I, ATTACHMENT, head.link, head.direction, head.window, head.digest) == head
    assert Entry.from_json(json.loads(json.dumps(head.to_json()))) == head
    assert live.to_json("s0/uplink") == {"link": "s0/uplink", "window": 0, "period": 0, "block": 0,
                                         "signature": live.signature.hex()}


def test_a_period_issuer_is_one_attachments_of_one_period_length():
    with pytest.raises(ValueError, match="at least one link-window"):
        PeriodIssuer(_Refusing(), ATTACHMENT, LINK, 0)
    with pytest.raises(ValueError, match="an attachment is 32 bytes"):
        PeriodIssuer(_Refusing(), ATTACHMENT[:16], LINK, M)


class _Refusing:
    """A party key whose leaf state is held elsewhere: it refuses every signature."""

    lower = bytes(16 + SMALL.N)

    def period_root(self, *args):
        raise lms.StateError("held")

    def liveness(self, *args):
        raise lms.StateError("held")


def test_a_signing_refusal_fails_closed_and_is_listed():
    issuer = PeriodIssuer(_Refusing(), ATTACHMENT, LINK, 2, coins=coins())
    certify(issuer, 0)
    assert issuer.live(0) is None
    certify(issuer, 1)
    assert issuer.live(1) is None
    assert issuer.refused == [("liveness", 0), ("liveness", 1), ("root", 0)] and issuer.closed == []
    assert issuer.close() == []
