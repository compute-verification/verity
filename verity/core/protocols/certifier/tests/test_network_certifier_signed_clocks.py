"""The certifier's signed clock sync, salted as a record digest is (`certificate.CertifiedClock`,
`Issuer.salted_clock`), read through its opening (`certificate.SignedClocks`): a window has a sync only from one
opening that names it and holds, and each refused opening is named with why: a forged signature, an altered digest, a
reused salt, a wrong window, another key, session or link, an ingress or a record certificate, a record's opening, and
two syncs for one window. The issuer signs no clock under a salt an earlier signed one took."""

import hashlib
from dataclasses import replace
from fractions import Fraction

import pytest

from verity.ml.commitments.identity import tagged_sha512
from verity.core.protocols.certifier.certificate import CLOCK_TAG, SALT_BYTES, CertifiedClock, Direction, Issuer, \
    SignedClocks, SignerRefused, clock_commitment, clock_digest, parameters_digest
from verity.core.protocols.certifier.grid import Certifier, GridParams
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet

import signers

LINK, WINDOWS = "responses", 4
P = GridParams(buckets=8, allocation=3, frame_bytes=1)
SYNC_SET = SyncSet((1, 2), (2, 4))
SYNCS = [ClockSync(1, 2), ClockSync(2, 4), ClockSync(2, 2), ClockSync(1, 4)]
INSTANT, ONE = ClockSync(0, 1), ClockSync(1, 1)


def _egress():
    """One egress window's record of `LINK`: a frame in its first bucket."""
    cert = Certifier(LINK, P)
    cert.enqueue(b"e")
    for _ in range(P.buckets):
        cert.tick()
    return cert.close_window()


class Refusing:
    """A signer whose leaf state refuses the session's start and every leaf."""

    def __init__(self, signer) -> None:
        self.key_id = signer.key_id

    def start(self, session: bytes) -> None:
        raise SignerRefused("the session's start is refused")

    def sign(self, message: bytes) -> bytes:
        raise SignerRefused("the leaf is refused")


def test_the_clock_commitment_is_the_syncs_clock_digest_salted_as_a_record_digest_is():
    assert clock_digest(ClockSync(3, Fraction(2, 4))) == tagged_sha512(
        CLOCK_TAG, (3).to_bytes(8, "big") + (1).to_bytes(8, "big") + (2).to_bytes(8, "big"))
    digests = {clock_digest(s) for s in [*SYNCS, INSTANT, ONE]}
    assert len(digests) == len(SYNCS) + 2
    with pytest.raises(ValueError):
        clock_digest(ClockSync(1 << 64, 1))
    salt = bytes(range(SALT_BYTES))
    assert clock_commitment(salt, SYNCS[0]) == hashlib.sha512(b"verity/net-cert/v1" + salt
                                                              + clock_digest(SYNCS[0])).digest()
    assert clock_commitment(salt, SYNCS[0]) != clock_commitment(bytes(SALT_BYTES), SYNCS[0])
    with pytest.raises(ValueError, match="salt is 64 bytes"):
        clock_commitment(salt[:-1], SYNCS[0])


@pytest.fixture
def signed(tmp_path):
    """An issuer, its key and its honest openings of `SYNCS`, one an egress window of `LINK`."""
    issuer = Issuer(signers.signer(tmp_path), signers.SESSION)
    return issuer, issuer.signer.public_key(), [issuer.salted_clock(LINK, w, s) for w, s in enumerate(SYNCS)]


def _source(key, openings, session=signers.SESSION):
    return SignedClocks(key, session, openings)


def _name(c, window=None):
    x = c.certificate
    return f"{x.link} window {x.window if window is None else window}'s {x.direction.value} clock " \
           f"(key {x.key_id.hex()[:16]}…)"


def test_a_window_has_a_sync_only_from_one_valid_opening_naming_it(signed, tmp_path):
    """`SignedClocks.syncs` gives a window the sync of the openings that name the link and the window and hold under
    the verifier's key and session, and stops at the first window without exactly one such sync. The same opening
    twice is one."""
    issuer, key, honest = signed

    def syncs(openings, session=signers.SESSION):
        return _source(key, openings, session).syncs(LINK)

    assert syncs(honest) == tuple(SYNCS) and syncs(honest[::-1]) == tuple(SYNCS) and syncs(honest + honest) == \
        tuple(SYNCS) and _source(key, honest + honest).refused == ()
    assert syncs(honest, signers.OTHER_SESSION) == () and syncs([]) == ()
    assert _source(key, honest).syncs("requests") == ()
    assert syncs(honest[1:]) == () and syncs([honest[0], *honest[2:]]) == (SYNCS[0],)
    assert syncs([*honest, issuer.salted_clock(LINK, 0, SYNCS[0])]) == tuple(SYNCS)


def _forged(issuer, honest):
    c = honest[0]
    return replace(c, certificate=replace(c.certificate, signature=bytes(len(c.certificate.signature))))


def _altered(issuer, honest):
    c = honest[0]
    return replace(c, certificate=replace(c.certificate, digest=bytes(64)))


def _swapped(issuer, honest):
    return replace(honest[0], sync=SYNCS[1])


def _relabelled(issuer, honest):
    c = honest[0]
    return replace(c, certificate=replace(c.certificate, window=WINDOWS))


def _reused(issuer, honest):
    """Window 0's salt again, under a certificate the certifier really signed for window 1's sync."""
    salt = honest[0].salt
    return CertifiedClock(SYNCS[1], salt, issuer.sign(LINK, Direction.EGRESS, 1, clock_commitment(salt, SYNCS[1])))


UNVERIFIED = "its signature doesn't verify under the key in its session for the link-window it names"
UNOPENED = "its sync and salt don't open the salted digest it signs"


@pytest.mark.parametrize("make, says", [
    (_forged, UNVERIFIED),
    (_altered, UNOPENED),
    (_swapped, UNOPENED),
    (_relabelled, UNVERIFIED),
    (_reused, f"its salt is also {LINK} window 0's, under which two windows of one sync sign one digest"),
])
def test_each_refused_opening_is_named_with_why_and_gives_no_sync(signed, make, says):
    """Window 0's opening in place of its own with a forged signature, an altered digest, a sync swapped after
    signing, or a wrong window (relabelled, so the message it signs is another window's); or beside the honest ones,
    an opening of window 1 under window 0's salt, which the certifier really signed. Each is refused by name with why,
    the honest opening whose salt it reuses too, and window 0 has no sync, so the link has none."""
    issuer, key, honest = signed
    bad = make(issuer, honest)
    reused = make is _reused
    source = _source(key, [*honest, bad] if reused else [bad, *honest[1:]])
    want = [f"{_name(bad)}: {says}"]
    if reused:
        want.insert(0, f"{_name(honest[0])}: its salt is also {LINK} window 1's, under which two windows of one sync "
                       "sign one digest")
    assert source.refused == tuple(want) and source.syncs(LINK) == ()
    assert bad.verify(key, signers.SESSION) is reused


def test_another_key_session_link_or_form_is_refused_by_name(signed, tmp_path):
    """An opening signed by another certifier, in another session, for another link, an ingress certificate, a record
    certificate opened as a clock, a record's opening itself (`CertifiedRecord`) and two syncs signed for one window:
    each refused with why."""
    issuer, key, honest = signed
    (tmp_path / "other").mkdir()
    other = Issuer(signers.signer(tmp_path / "other", seed=b"stranger"), signers.SESSION)
    foreign = other.salted_clock(LINK, 0, SYNCS[0])
    elsewhere = issuer.salted_clock("requests", 0, SYNCS[0])
    salt = bytes(range(SALT_BYTES))
    ingress = CertifiedClock(SYNCS[0], salt, issuer.sign(LINK, Direction.INGRESS, 0, clock_commitment(salt, SYNCS[0])))
    record = issuer.certify(_egress(), Direction.EGRESS, P, parameters_digest(P, SYNC_SET))
    as_clock = CertifiedClock(SYNCS[0], record.salt, record.certificate)
    second = issuer.salted_clock(LINK, 0, SYNCS[1])
    source = _source(key, [foreign, ingress, as_clock, record, *honest, second])
    assert source.refused == (
        f"{_name(foreign)}: another key's, not {key.key_id.hex()[:16]}…",
        f"{_name(ingress)}: it isn't an egress certificate, and a clock is an egress window's",
        f"{_name(as_clock)}: its sync and salt don't open the salted digest it signs",
        "a CertifiedRecord: not a salted clock certificate's opening (certificate.CertifiedClock), so it gives no sync",
        f"{LINK} window 0: 2 signed syncs, so it has none",
    )
    assert source.syncs(LINK) == ()
    assert _source(key, honest, signers.OTHER_SESSION).refused == tuple(
        f"{_name(c)}: its signature doesn't verify under the key in its session for the link-window it names"
        for c in honest)
    assert _source(key, [elsewhere, *honest]).syncs(LINK) == tuple(SYNCS)
    assert _source(key, [elsewhere]).syncs("requests") == (SYNCS[0],)


def test_the_issuer_signs_no_clock_under_a_salt_it_drew_for_one_before(tmp_path):
    """Coins that repeat give the first signed clock its salt and refuse the second, which signs nothing; a clock the
    signer refused signed nothing, so its salt isn't taken; a record's salt isn't a clock's, so the record certificates
    of those coins are signed as before."""
    issuer = Issuer(signers.signer(tmp_path), signers.SESSION, coins=lambda n: b"\x5a" * n)
    signer, issuer.signer = issuer.signer, Refusing(issuer.signer)
    assert issuer.salted_clock(LINK, 0, SYNCS[0]) is None and issuer.clock_salts == set()
    issuer.signer = signer
    first = issuer.salted_clock(LINK, 0, SYNCS[0])
    assert first.salt == b"\x5a" * SALT_BYTES and first.verify(issuer.signer.public_key(), signers.SESSION)
    leaves = len(signers.used(issuer.signer))
    with pytest.raises(ValueError, match="gave an earlier clock certificate's salt again"):
        issuer.salted_clock(LINK, 1, SYNCS[0])
    assert len(signers.used(issuer.signer)) == leaves and issuer.refused == [(LINK, Direction.EGRESS, 0)]
    record = issuer.certify(_egress(), Direction.EGRESS, P, parameters_digest(P, SYNC_SET))
    assert record.salt == first.salt and record.verify(issuer.signer.public_key(), signers.SESSION)
    with pytest.raises(ValueError, match="gave 63 bytes for a 64-byte salt"):
        Issuer(issuer.signer, signers.SESSION, coins=lambda n: bytes(n - 1)).salted_clock(LINK, 0, SYNCS[0])
