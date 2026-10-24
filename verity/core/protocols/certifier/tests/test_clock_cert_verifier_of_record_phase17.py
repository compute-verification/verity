"""REDTEAM phase 17, Target 1 (#3881, CERT-VARLEN): the clock certificate as the verifier of record's clock source,
exercised end to end through an advised link (`SignedRun.link`), not only at `SignedClocks.syncs`. The PR's own test
covers the per-certificate refusals at the source; these confirm the same refusals drive a whole advised link to
`[False] * WINDOWS`, and that an invalid certificate delivered **beside** a valid one for a window is dropped, not a
grief that refuses the window.

Every certificate here is signed by the run's own trusted certifier: an attacker who only delivers them (the prover)
cannot forge one, so the only certificates it can place for a window are ones the certifier actually signed."""

from dataclasses import replace

import pytest

from verity.core.protocols.certifier import advised as AD, service as SVC
from verity.core.protocols.certifier.certificate import ClockCertificate, Direction, Issuer, parameters_digest
import signers
import standin
from advised_traffic import LINK, P, SYNC_SET, SYNCS, WINDOWS, serve
from test_network_certifier_advised_signed_clocks import SignedRun, _signed


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(SVC.PF, "verify", standin.lean)


def test_an_honest_run_through_the_verifier_of_record_is_accepted(tmp_path):
    """The baseline: each egress window's sync signed by the run's certifier, every window accepted."""
    assert SignedRun(tmp_path / "honest", _signed(SYNCS)).link(serve(1)) == [True] * WINDOWS


def test_a_forged_certificate_beside_the_valid_one_for_a_window_is_dropped_not_a_grief(tmp_path):
    """An invalid certificate for window 0 (a zeroed signature) delivered alongside the valid one is filtered before the
    per-window count, so window 0 keeps its one valid sync and the run is still accepted. A deliverer can't grief a
    window into refusal by padding it with junk."""
    def sign(issuer):
        honest = [issuer.clock(LINK, w, s) for w, s in enumerate(SYNCS)]
        c0 = honest[0].certificate
        forged0 = ClockCertificate(SYNCS[0], replace(c0, signature=bytes(len(c0.signature))))
        return [*honest, forged0]

    assert SignedRun(tmp_path / "forged_beside", sign).link(serve(1)) == [True] * WINDOWS


def test_a_duplicate_of_the_same_signed_sync_for_a_window_dedups_and_is_accepted(tmp_path):
    """Two certificates for window 0 carrying the identical signed sync collapse in the set, so the window keeps exactly
    one sync and the run is accepted (no ambiguity to fail closed on)."""
    def sign(issuer):
        honest = [issuer.clock(LINK, w, s) for w, s in enumerate(SYNCS)]
        return [*honest, issuer.clock(LINK, 0, SYNCS[0])]

    assert SignedRun(tmp_path / "dup_same", sign).link(serve(1)) == [True] * WINDOWS


def test_two_different_signed_syncs_for_one_window_refuse_it_and_every_window_after(tmp_path):
    """Two certificates for window 0 carrying different signed syncs give the window two syncs, so it has none: the
    whole advised link is refused. Producing the second valid certificate needs the certifier's key; the prover has no
    way to make one."""
    def sign(issuer):
        honest = [issuer.clock(LINK, w, s) for w, s in enumerate(SYNCS)]
        return [*honest, issuer.clock(LINK, 0, SYNCS[1])]

    assert SignedRun(tmp_path / "two_syncs", sign).link(serve(1)) == [False] * WINDOWS


def test_clock_certificates_signed_for_another_link_drive_no_floor(tmp_path):
    """The verifier of record is built for LINK ('responses'); clock certificates the certifier signed for another link
    name that link, so `SignedClocks.syncs(LINK)` is empty and every advised window is refused. A stand-in or
    wrong-context source can't sneak its syncs into this link's floors."""
    assert SignedRun(tmp_path / "cross_link", _signed(SYNCS, link="requests")).link(serve(1)) == [False] * WINDOWS


def test_a_record_certificate_delivered_as_a_windows_clock_is_refused_end_to_end(tmp_path):
    """Window 0's only clock certificate is a record certificate wrapped as a `ClockCertificate`: its signed digest is
    the salted record digest, not the clock digest, so it fails `ClockCertificate.verify`, window 0 has no sync, and the
    advised link is refused."""
    def sign(issuer):
        honest = [issuer.clock(LINK, w, s) for w, s in enumerate(SYNCS)]
        record = issuer.certify(serve(1).egress[0], Direction.EGRESS, P, parameters_digest(P, SYNC_SET))
        return [ClockCertificate(SYNCS[0], record.certificate), *honest[1:]]

    assert SignedRun(tmp_path / "record_as_clock", sign).link(serve(1)) == [False] * WINDOWS


def test_a_missing_middle_window_refuses_the_whole_link_not_only_windows_after_it(tmp_path):
    """No certificate for window 1: `SignedClocks.syncs` stops at window 0. Because every window's floor reads the
    certified sync of each output's release window (`_floor_buckets` is link-wide), an output released at or after the
    gap has 'no sync' and raises, so even window 0's view is refused. A single missing certified sync fails the whole
    link closed, not just the windows after the gap."""
    def sign(issuer):
        certs = [issuer.clock(LINK, w, s) for w, s in enumerate(SYNCS)]
        return [certs[0], *certs[2:]]

    assert SignedRun(tmp_path / "gap_w1", sign).link(serve(1)) == [False] * WINDOWS
