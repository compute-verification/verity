"""Phase 17 red-team (#3881 CERT-VARLEN x #3862 N2-ZK), Target 5b: #3881's default clock source refuses an advised link
unless a clock certificate is given, and the stand-in clock the advised driver (#3862) supplies is a separate opt-in
that can never masquerade as the certificate source of record.

`LinkWindows.clock_source` prefers the certifier's clock certificates (`advised.SignedClocks`), else the stand-in
`clocks`, else None, which refuses every advised window (`LinkWindows._view`). So:

* With no clock source at all -- no certificates and no stand-in -- the whole advised link is refused, in every window
  (fail closed).
* The stand-in `clocks` (a `ClockSource` by link, the shape `#3862`'s `CertifierClocks`/`StandInClocks` carry) is the
  explicit rehearsal opt-in: given it, the honest link is accepted, under a `clock_source` that is the stand-in, not a
  `SignedClocks`.
* The certifier's clock certificates are the accepted default of record: given them, the honest link is accepted under
  a `SignedClocks`.
* The two are separate fields: passing both is an error, and the certificate source is always a `SignedClocks` built
  from clock certificates, never a stand-in `ClockSource`, so the driver's stand-in can't be read as the default.
"""

from dataclasses import dataclass

import pytest

from verity.core.protocols.certifier import advised as AD, service as SVC
from verity.core.protocols.certifier.schedule import ClockSync
from verity.core.protocols.certifier.service import LinkWindows

import signers
import standin
from advised_traffic import LINK, P, SYNC_SET, SYNCS, WINDOWS, serve
from test_network_certifier_advised import Run
from test_network_certifier_advised_signed_clocks import SignedRun, _signed


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(SVC.PF, "verify", standin.lean)


@dataclass(frozen=True)
class LogClock:
    """A stand-in clock source of the shape #3862's driver reads from ``proxy.json.gz`` (`CertifierClocks`): the syncs
    the certifier ran, by link, and no certificate behind them."""

    ran: tuple[ClockSync, ...]

    def syncs(self, link: str) -> tuple[ClockSync, ...]:
        return self.ran if link == LINK else ()


def test_an_advised_link_with_no_clock_source_is_refused_in_every_window(tmp_path):
    """No certificates and no stand-in: `clock_source` is None and every window is refused (fail closed)."""
    run = Run(tmp_path, clocks=None)
    assert run.lw.clock_source is None
    assert run.link(serve(1)) == [False] * WINDOWS


def test_the_drivers_stand_in_clock_source_is_an_opt_in_that_is_accepted(tmp_path):
    """Given the stand-in `clocks` the advised driver supplies, the honest link is accepted, under a `clock_source`
    that is the stand-in itself, not a `SignedClocks`."""
    run = Run(tmp_path, clocks=LogClock(tuple(SYNCS)))
    assert type(run.lw.clock_source) is LogClock and run.lw.clock_source.syncs(LINK) == tuple(SYNCS)
    assert run.link(serve(1)) == [True] * WINDOWS


def test_the_clock_certificate_is_the_accepted_default_of_record(tmp_path):
    """Given the certifier's clock certificates, the honest link is accepted under a `SignedClocks`."""
    run = SignedRun(tmp_path / "signed", _signed(SYNCS))
    assert type(run.lw.clock_source) is AD.SignedClocks
    assert run.link(serve(1)) == [True] * WINDOWS


def test_the_stand_in_and_the_certificate_source_are_separate_fields(tmp_path):
    """A link takes the stand-in `clocks` or the certificates, not both; with neither the source is None; and the
    certificate source is always a `SignedClocks` with `clocks` unset, so the driver's stand-in can't be read as the
    default of record."""
    key = signers.signer(tmp_path).public_key()
    args = (LINK, P, SYNC_SET, key, "r-clocks", signers.SESSION)
    with pytest.raises(ValueError, match="not both"):
        LinkWindows(*args, clocks=LogClock(()), clock_certificates=())
    assert LinkWindows(*args).clock_source is None
    run = SignedRun(tmp_path / "signed2", _signed(SYNCS))
    assert type(run.lw.clock_source) is AD.SignedClocks and run.lw.clocks is None
