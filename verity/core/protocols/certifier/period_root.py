"""The network certifier as an auditor party on device-attest: every digest it certifies on a link goes under one signed
root a certification period, and its one public output a sampling block is the block's liveness record
(`verity.core.attest`).

The certifier holds a lower HSS key its device's identity certified for it in the deployment (`lms.Identity.issue` for
`PARTY`, `lms.PartyKey`), and signs no certificate alone. `PeriodIssuer` takes `certificate.Issuer`'s place: it salts and
heads a record (`certify`), salts a sync (`salted_clock`) and takes a 64-byte digest (`sign`, which `warden_rows.attest`
calls with a cells head's digest) exactly as `Issuer` does, and each digest becomes one record of the period: the message
its certificate would have signed, `certificate.certificate_message(I, attachment, link, direction, window, digest)`, with
``I`` the lower key's identifier and the attachment the certifier's 32-byte session id. The record binds what a
certificate bound (the key, the attachment, the link, its end, the link-window and the hiding digest), so a proof that
reads one learns which link-window's digest it is from the record alone, at whatever position it sits, and a record of
one link-window opens as another's only under a SHA-512 collision. What a certificate's signature was, the record's
opening to its period's signed root is.

**Blocks and periods.** Certification period ``p`` covers the link-windows (sampling blocks) ``first + p·m`` to
``first + (p + 1)·m − 1``, ``m`` = `PeriodIssuer.windows`. `PeriodIssuer.live` closes link-window ``w``: its records, in
the order they were certified, are appended as block ``b = (w − first) mod m`` of its period (`attest.Period.block`), and
it returns the block's `Liveness`, whose signature is the link-window's one output in the clear: "block b of period p was
certified", with no content (`lms.liveness_message`). Once block ``m − 1`` is live the period's root is signed
(`attest.Period.close`), and the `attest.SignedPeriod` (its records, root and two-level signature) goes to
`PeriodIssuer.closed`, for the prover alone; a link-window of a later period, or `PeriodIssuer.close` at a run's end,
closes an earlier period early. A link-window `live` never closes (one in which the certifier refused a connection,
`warden`'s "Admission", or one a run ended in) has no liveness record and no record in any period: it is certified
nowhere. A signing refusal (`lms.StateError`: a leaf state missing, behind or held) fails closed and is listed in
`PeriodIssuer.refused`, a block's as ``("liveness", w)`` and a period's root as ``("root", p)``: a refused block's records
are in its period (`attest.Period.block` appends them before it signs), but with no liveness record none of them reads as
certified, and a refused root certifies none of its period's.

`refusal` is the period check in the clear, a rehearsal's reference: the closed period's root is its records' and its
signature verifies under the device's identity for `PARTY`, every liveness record the caller holds verifies, and every
entry the caller reads is a record of the period in a block with a liveness record. A block without one fails only its
own entries, not its period. A period's proof runs the same check on its witness (`MemoryChallenger.Attest.check`,
whose soundness `Properties.Attest.PeriodSigned` states), and the verifier reads only the liveness records in the clear.
"""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from verity.core import attest
from verity.core.protocols.certifier.certificate import (DIGEST_BYTES, Certificate, Direction, Issuer, SignerRefused,
                                                         certificate_message)
from verity.core.protocols.certifier.grid import GridParams, Record
from verity.core.protocols.certifier.schedule import ClockSync
from verity.ml.crypto import lms

__all__ = ["PARTY", "Entry", "Liveness", "PartyKey", "PeriodClock", "PeriodIssuer", "PeriodRecord", "refusal"]

#: the network certifier's party name in a deployment, which its device's identity certifies its lower key for
PARTY = "network-certifier"


class PartyKey(Protocol):
    """A party's lower key as `PeriodIssuer` uses it (`lms.PartyKey`): `attest.Signer`'s two signatures, and the lower public
    key ``I ‖ T[1]`` whose identifier every record binds."""

    lower: bytes

    def period_root(self, attachment: bytes, link: str, period: int, count: int, root: bytes): ...

    def liveness(self, attachment: bytes, link: str, period: int, block: int): ...


@dataclass(frozen=True, slots=True)
class Entry:
    """One digest the certifier certified on a link-window, as a record of its certification period: the fields a
    certificate's public part had, without a signature, and ``record``, their `certificate_message` under the lower key's
    identifier in the attachment."""

    link: str
    direction: Direction
    window: int
    digest: bytes
    record: bytes


@dataclass(frozen=True, slots=True)
class PeriodRecord:
    """`certificate.CertifiedRecord`'s place: the record head and the salt the certifier drew for it (witness), and the
    entry of its salted digest (`certificate.certificate_digest`)."""

    head: object
    salt: bytes
    entry: Entry


@dataclass(frozen=True, slots=True)
class PeriodClock:
    """`certificate.CertifiedClock`'s place: the sync and its salt (witness), and the entry of its salted clock digest
    (`certificate.clock_commitment`)."""

    sync: ClockSync
    salt: bytes
    entry: Entry


@dataclass(frozen=True, slots=True)
class Liveness:
    """Link-window ``window`` closed as block ``block`` of period ``period``: its liveness record (``signature``, the
    bytes of an `lms.HssSignature`), the one thing about it the verifier reads in the clear, and the entries it appended
    to the period, in order."""

    window: int
    period: int
    block: int
    signature: bytes
    entries: tuple[Entry, ...]


class _Records:
    """The `certificate.Signer` that `PeriodIssuer` runs its `Issuer` on: it signs nothing, and keeps the message it was
    last asked to sign, a certificate's (`certificate_message`), which is the period's record. It starts only the
    attachment."""

    def __init__(self, key_id: bytes, attachment: bytes) -> None:
        self.key_id, self.attachment, self.message = key_id, attachment, b""

    def start(self, session: bytes) -> None:
        if session != self.attachment:
            raise SignerRefused("a period issuer certifies in its own attachment only")

    def sign(self, message: bytes) -> bytes:
        self.message = message
        return b""


@dataclass
class PeriodIssuer:
    """The certifier's signing step on device-attest, in place of `certificate.Issuer` (the module docstring): the
    link-windows ``first``, ``first + 1``, … of ``attachment`` on ``link``, in certification periods of ``windows``
    link-windows, under the party key ``key``. Each record's and each clock's salt is fresh bytes of ``coins``, as
    `Issuer`'s are. On an `lms.PartyKey` it first records the attachment as started on the lower key's leaf state, and
    raises `SignerRefused` when the state has started it already."""

    key: PartyKey
    attachment: bytes
    link: str
    windows: int
    first: int = 0
    coins: Callable[[int], bytes] = field(default=secrets.token_bytes, repr=False)
    closed: list[attest.SignedPeriod] = field(default_factory=list)
    refused: list[tuple[str, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if type(self.windows) is not int or self.windows < 1:
            raise ValueError(f"a certification period is at least one link-window, not {self.windows!r}")
        if len(self.attachment) != lms.SESSION_BYTES:
            raise ValueError(f"an attachment is {lms.SESSION_BYTES} bytes, not {len(self.attachment)}")
        if isinstance(self.key, lms.PartyKey):
            # As `Issuer` starts its session: no second issuer (a restarted certifier among them) signs a second root
            # for one of the attachment's periods, of which the prover would pick one.
            try:
                with lms.LeafState(self.key.directory / lms.LOWER_STATE, self.key.lower, self.key.params.H,
                                   wait=True) as state:
                    state.start(bytes(self.attachment))
            except lms.StateError as e:
                raise SignerRefused(str(e)) from None
        self._signer = _Records(bytes(self.key.lower[:16]), bytes(self.attachment))
        self._issuer = Issuer(self._signer, bytes(self.attachment), coins=self.coins)
        self._pending: dict[int, list[Entry]] = {}
        self._period: attest.Period | None = None
        self._last = self.first - 1
        self._lock = threading.Lock()

    @property
    def session(self) -> bytes:
        """The attachment, under the name `warden_rows.Cells.of` reads an issuer's."""
        return self.attachment

    def period_of(self, window: int) -> tuple[int, int]:
        """Link-window ``window``'s period and its block in it."""
        return divmod(window - self.first, self.windows)

    def _open(self, window: int) -> None:
        if type(window) is not int or window < self.first:
            raise ValueError(f"link-window {window!r} is before the attachment's first, {self.first}")
        if window <= self._last:
            raise ValueError(f"link-window {window} is not after link-window {self._last}, already live")

    def _keep(self, certificate: Certificate | None) -> Entry:
        if certificate is None:
            raise SignerRefused("the period issuer's own issuer refused, which it never does")
        entry = Entry(certificate.link, certificate.direction, certificate.window, certificate.digest,
                      self._signer.message)
        self._pending.setdefault(entry.window, []).append(entry)
        return entry

    def certify(self, record: Record, direction: Direction, params: GridParams, parameters: bytes, *,
                deadline: bool = True) -> PeriodRecord:
        """`Issuer.certify`: the record's head salted, its salted digest an entry of the link-window."""
        with self._lock:
            self._open(record.window)
            c = self._issuer.certify(record, direction, params, parameters, deadline=deadline)
            return PeriodRecord(c.head, c.salt, self._keep(c.certificate))

    def salted_clock(self, link: str, window: int, sync: ClockSync) -> PeriodClock:
        """`Issuer.salted_clock`: the sync's clock digest salted (no salt twice), an egress entry of the link-window."""
        with self._lock:
            self._open(window)
            c = self._issuer.salted_clock(link, window, sync)
            if c is None:
                raise SignerRefused("the period issuer's own issuer refused, which it never does")
            return PeriodClock(c.sync, c.salt, self._keep(c.certificate))

    def sign(self, link: str, direction: Direction, window: int, digest: bytes) -> Entry:
        """`Issuer.sign`: the 64-byte ``digest``, an entry of the link-window."""
        with self._lock:
            self._open(window)
            if len(digest) != DIGEST_BYTES:
                raise ValueError("a signed digest is 64 bytes")
            return self._keep(self._issuer.sign(link, direction, window, digest))

    def live(self, window: int) -> Liveness | None:
        """Close link-window ``window`` as its period's block (the module docstring) and return its liveness record, or
        None when the party key refused to sign it. Entries of earlier link-windows that never went live are dropped:
        they are in no period. The link-window's period closes after its last block."""
        with self._lock:
            self._open(window)
            period, block = self.period_of(window)
            if self._period is not None and self._period.period != period:
                self._close()
            if self._period is None:
                self._period = attest.Period(self.key, bytes(self.attachment), self.link, period)
            entries = tuple(self._pending.pop(window, ()))
            for w in [w for w in self._pending if w < window]:
                del self._pending[w]
            self._last = window
            try:
                signature = self._period.block(block, [e.record for e in entries])
            except lms.StateError:
                self.refused.append(("liveness", window))
                return None
            finally:
                if block == self.windows - 1:
                    self._close()
            return Liveness(window, period, block, signature, entries)

    def _close(self) -> None:
        period, self._period = self._period, None
        try:
            self.closed.append(period.close())
        except lms.StateError:
            self.refused.append(("root", period.period))

    def close(self) -> list[attest.SignedPeriod]:
        """End the run: close the open period, whatever blocks it has, and drop the entries of link-windows never made
        live. Returns every period closed."""
        with self._lock:
            if self._period is not None:
                self._close()
            self._pending.clear()
            return list(self.closed)


def refusal(closed: attest.SignedPeriod, liveness: Mapping[int, bytes], entries: Sequence[Entry], *, identity: bytes,
            deployment: bytes, attachment: bytes, link: str, period: int, windows: int, first: int = 0,
            params: lms.Params = lms.DEPLOYED) -> str | None:
    """Why certification period ``period`` of ``attachment`` on ``link`` fails in the clear, or None when it holds:
    ``closed`` is that period, its root is its records' and its two-level signature verifies under the device's
    ``identity`` for `PARTY` in the deployment whose hash is ``deployment``; each liveness record held (``liveness[b]``,
    its bytes, for block ``b`` of the period's ``windows``) verifies so; and each of ``entries`` is a link-window of the
    period's whose block has a liveness record, its record the `certificate_message` of its fields under the root's lower
    key in the attachment, and a record of the period. A rehearsal's check; a deployment's verifier reads the liveness
    records alone, and a proof the rest."""
    if (closed.attachment, closed.link, closed.period) != (bytes(attachment), link, period):
        return f"another period's: attachment {closed.attachment.hex()[:16]}…, link {closed.link!r}, period {closed.period}"
    if closed.root != attest.root(closed.records):
        return "its root isn't its records'"
    unsigned = "its root's signature doesn't verify under the device's identity"
    try:
        sig = lms.HssSignature.decode(params, closed.signature)
        message = lms.period_root_message(sig.lower[:16], deployment, PARTY, closed.attachment, closed.link,
                                          closed.period, closed.count, closed.root)
        signed = lms.hss_verify(params, identity, deployment, PARTY, sig, message)
    except ValueError as err:
        return f"{unsigned}: {err}"
    if not signed:
        return unsigned
    for b in sorted(set(liveness) - set(range(windows))):
        return f"a liveness record for block {b}, which a period of {windows} hasn't"
    for b in sorted(liveness):
        dead = f"block {b}'s liveness record doesn't verify under the device's identity"
        try:
            live = lms.verify_liveness(params, identity, deployment, PARTY, closed.attachment, closed.link, period, b,
                                       lms.HssSignature.decode(params, liveness[b]))
        except ValueError as err:
            return f"{dead}: {err}"
        if not live:
            return dead
    records, lo = set(closed.records), first + period * windows
    for e in entries:
        what = f"{e.link} {e.direction.value} link-window {e.window}"
        if not lo <= e.window < lo + windows:
            return f"{what}: not a link-window of the period"
        if e.window - lo not in liveness:
            return f"{what}: its block, {e.window - lo}, has no liveness record"
        unbound = f"{what}: its record isn't its fields' under the root's party key"
        try:
            record = certificate_message(sig.lower[:16], closed.attachment, e.link, e.direction, e.window, e.digest)
        except ValueError as err:
            return f"{unbound}: {err}"
        if e.record != record:
            return unbound
        if e.record not in records:
            return f"{what}: not a record of the period"
    return None
