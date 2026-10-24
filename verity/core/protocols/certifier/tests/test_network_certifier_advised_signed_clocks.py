"""The certifier's signed clock sync (`certificate.ClockCertificate`, `Issuer.clock`) as an advised link's clock source
of record (`advised.SignedClocks`, `LinkWindows(clock_certificates=…)`): an honest link whose certifier signed each
egress window's sync is accepted; a window declaring a sync other than the one signed for it is refused, REDTEAM phase
11's δ below the certified one among them; and a window whose sync is unsigned or wrongly signed (another key, session,
link, window or direction, a sync swapped after signing, a record certificate, a forged signature, two syncs for one
window) has no sync, which refuses it and every window after."""

from dataclasses import replace
from fractions import Fraction

import pytest

from verity.ml.commitments.identity import tagged_sha512
from verity.core.protocols.certifier import advice as A, advised as AD, service as SVC
from verity.core.protocols.certifier.certificate import CLOCK_TAG, ClockCertificate, Direction, Issuer, clock_digest, \
    parameters_digest
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.certifier.service import LinkWindows

import signers
import standin
from advised_traffic import LINK, P, SYNC_SET, SYNCS, WINDOWS, held, serve
from test_network_certifier_advised import Run

T, R = 8, 2
#: phase 11's case: one request of 2 frames certified in bucket 3, its output released in bucket 3 too
FRAMING = tuple(-(-n // 2) for n in range(1, 9))
LOOSE = SyncSet((0, 1), (1,))
INSTANT, ONE = ClockSync(0, 1), ClockSync(1, 1)


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(SVC.PF, "verify", standin.lean)


class SignedRun(Run):
    """`Run` whose verifier of record takes the syncs from the certifier's clock certificates alone, `sign(issuer)`
    signed by the run's own certifier."""

    def __init__(self, tmp_path, sign) -> None:
        tmp_path.mkdir()
        super().__init__(tmp_path, clocks=None)
        self.lw = replace(self.lw, clock_certificates=tuple(sign(self.issuer)))
        self.stream = self.lw.stream(budget=8, delta=2.0 ** -40, challenger="challenger-test",
                                     coins=lambda n: b"\x07" * n, clock=lambda: 0.0)


def _signed(syncs, link=LINK):
    return lambda issuer: [issuer.clock(link, w, s) for w, s in enumerate(syncs)]


def test_the_clock_digest_is_the_syncs_tagged_hash_in_lowest_terms():
    assert clock_digest(ClockSync(3, Fraction(2, 4))) == tagged_sha512(
        CLOCK_TAG, (3).to_bytes(8, "big") + (1).to_bytes(8, "big") + (2).to_bytes(8, "big"))
    digests = {clock_digest(s) for s in [*SYNCS, INSTANT, ONE]}
    assert len(digests) == len(SYNCS) + 2
    with pytest.raises(ValueError):
        clock_digest(ClockSync(1 << 64, 1))


def test_an_honest_link_is_accepted_on_its_certifiers_signed_syncs(tmp_path):
    """Each egress window's sync signed beside its record, the verifier of record accepts every window, on the clock
    source the link's own key and session give (`LinkWindows.clock_source`)."""
    run = SignedRun(tmp_path / "honest", _signed(SYNCS))
    assert type(run.lw.clock_source) is AD.SignedClocks and run.lw.clock_source.syncs(LINK) == tuple(SYNCS)
    assert run.link(serve(1)) == [True] * WINDOWS


def test_a_declared_sync_other_than_the_signed_one_is_refused(tmp_path):
    """The certifier signed each window at a δ one above the declared one: every window is refused. On phase 11's
    coincident release, δ = 0 declared is accepted where the certifier signed δ = 0, refused where it signed δ = 1."""
    slower = [ClockSync(s.delta + 1, s.rho) for s in SYNCS]
    assert SignedRun(tmp_path / "slower", _signed(slower)).link(serve(1)) == [False] * WINDOWS
    issuer = Issuer(signers.signer(tmp_path), signers.SESSION)
    key = issuer.signer.public_key()
    adv = A.Advice((A.RequestAdvice("a", 4, 1, "eos"),), (A.StepAdvice((A.Row(0, "prefill", 0, 4),)),), (3,))
    link = AD.LinkAdvice(adv, (2,))

    def view(signed):
        clocks = AD.SignedClocks(key, signers.SESSION, [issuer.clock(LINK, 0, signed)])
        return AD.view(link, 0, T, R, [INSTANT], LOOSE, FRAMING, clocks=clocks.syncs(LINK), **held(link))

    v = view(INSTANT)
    assert AD.reference(T, 4, 0, [0, 0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], v.released, v.floor, v.received)
    with pytest.raises(ValueError, match=r"declares ClockSync\(delta=0.*the certifier ran it under ClockSync\(delta=1"):
        view(ONE)


def test_an_unsigned_or_wrongly_signed_link_is_refused(tmp_path):
    """No clock certificates, or each signed by another certifier's key: every window is refused."""
    session = serve(1)
    assert SignedRun(tmp_path / "unsigned", lambda issuer: ()).link(session) == [False] * WINDOWS
    stranger = Issuer(signers.signer(tmp_path, seed=b"stranger"), signers.SESSION)
    assert SignedRun(tmp_path / "stranger", lambda issuer: _signed(SYNCS)(stranger)).link(session) == \
        [False] * WINDOWS


def test_a_window_has_a_sync_only_from_one_valid_certificate_naming_it(tmp_path):
    """`SignedClocks.syncs` gives a window the sync of the certificates that name the link, egress and the window and
    verify under the verifier's key and session, and stops at the first window without exactly one such sync."""
    for name in ("own", "other"):
        (tmp_path / name).mkdir()
    issuer = Issuer(signers.signer(tmp_path / "own"), signers.SESSION)
    key = issuer.signer.public_key()
    honest = [issuer.clock(LINK, w, s) for w, s in enumerate(SYNCS)]

    def syncs(certificates, session=signers.SESSION):
        return AD.SignedClocks(key, session, certificates).syncs(LINK)

    assert syncs(honest) == tuple(SYNCS) and syncs(honest[::-1]) == tuple(SYNCS) and syncs(honest + honest) == \
        tuple(SYNCS)
    assert syncs(honest, signers.OTHER_SESSION) == () and syncs([]) == ()
    assert AD.SignedClocks(key, signers.SESSION, honest).syncs("requests") == ()
    other = Issuer(signers.signer(tmp_path / "other"), signers.OTHER_SESSION)
    assert syncs([other.clock(LINK, 0, SYNCS[0]), *honest[1:]]) == ()
    assert syncs([issuer.clock("requests", 0, SYNCS[0]), *honest[1:]]) == ()
    assert syncs(honest[1:]) == () and syncs([honest[0], *honest[2:]]) == (SYNCS[0],)
    c = honest[0].certificate
    ingress = issuer.sign(LINK, Direction.INGRESS, 0, clock_digest(SYNCS[0]))
    swapped = ClockCertificate(SYNCS[1], c)
    forged = ClockCertificate(SYNCS[0], replace(c, signature=bytes(len(c.signature))))
    record = issuer.certify(serve(1).egress[0], Direction.EGRESS, P, parameters_digest(P, SYNC_SET))
    as_clock = ClockCertificate(SYNCS[0], record.certificate)
    for bad in (ClockCertificate(SYNCS[0], ingress), swapped, forged, as_clock):
        assert not bad.verify(key, signers.SESSION)
        assert syncs([bad, *honest[1:]]) == (), bad
    assert syncs([*honest, issuer.clock(LINK, 0, SYNCS[1])]) == ()
    assert syncs([*honest, issuer.clock(LINK, 0, SYNCS[0])]) == tuple(SYNCS)


def test_a_link_takes_clock_certificates_or_a_stand_in_source_not_both(tmp_path):
    key = signers.signer(tmp_path).public_key()
    args = (LINK, P, SYNC_SET, key, "r-clocks", signers.SESSION)
    with pytest.raises(ValueError, match="not both"):
        LinkWindows(*args, clocks=AD.SignedClocks(key, signers.SESSION, ()), clock_certificates=())
    assert LinkWindows(*args).clock_source is None
    assert LinkWindows(*args, clock_certificates=()).clock_source.syncs(LINK) == ()
