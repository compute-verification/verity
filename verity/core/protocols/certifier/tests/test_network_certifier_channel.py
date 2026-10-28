"""The certifier device keyed by a channel: a second channel kind, a toy meter with its own record and message, signs and
verifies through the device's shared path (its signer, session, salt, signed digest and signature check), and a link
certificate is the bytes it was before the device took channels."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import ClassVar

import pytest

from verity.ml.commitments.identity import tagged_sha512
from verity.ml.crypto import lms
from verity.core.protocols.certifier.certificate import (
    Certificate, ChannelCertificate, Direction, Issuer, LmsSigner, Link, certificate_digest, certificate_message,
    channel_message)
from verity.core.protocols.certifier.grid import COMPLETE, GridParams, Record, fill
from verity.core.protocols.certifier.schedule import ClockSync

import signers

SESSION, OTHER_SESSION = signers.SESSION, signers.OTHER_SESSION
TOY_TAG = b"verity/test-toy-certificate-message/v1"


def _field(b: bytes) -> bytes:
    return struct.pack(">Q", len(b)) + b


@dataclass(frozen=True)
class Meter:
    """A toy channel kind: a meter by name, its message framed as a link's under its own tag."""

    name: str

    kind: ClassVar[str] = "toy"

    @property
    def id(self) -> tuple[str]:
        return (self.name,)

    def message(self, key_id: bytes, session: bytes, window: int, digest: bytes) -> bytes:
        if len(session) != lms.SESSION_BYTES:
            raise ValueError("a certifier session id is 32 bytes")
        context = _field(session) + _field(self.name.encode()) + window.to_bytes(8, "big")
        return hashlib.sha512(_field(TOY_TAG) + key_id + _field(context) + _field(digest)).digest()


@dataclass(frozen=True)
class Reading:
    """The toy kind's record: a meter's readings over one window."""

    meter: str
    window: int
    joules: tuple[int, ...]

    def digest(self) -> bytes:
        return tagged_sha512("verity-test-toy/record/v1",
                             _field(self.meter.encode()) + b"".join(struct.pack(">Q", j) for j in self.joules))


def certify_reading(issuer: Issuer, reading: Reading) -> tuple[bytes, ChannelCertificate | None]:
    salt = issuer.draw_salt()
    return salt, issuer.sign_on(Meter(reading.meter), reading.window, certificate_digest(salt, reading.digest()))


def test_a_second_channel_kind_signs_and_verifies_through_the_shared_path(tmp_path):
    """A toy meter's reading is signed by the issuer that signs links, on one signer and session: its certificate
    verifies under the key in the session and opens to the reading under the salt, and fails moved to another window,
    meter, session, key or digest, or read as a link certificate. A refusal fails closed and lists the channel-window."""
    signer = signers.signer(tmp_path)
    key = signer.public_key()
    issuer = Issuer(signer, SESSION, coins=lambda n: b"\x05" * n)
    reading = Reading("pdu-3", 5, (120, 118, 131))
    salt, cert = certify_reading(issuer, reading)
    assert cert is not None and cert.channel == Meter("pdu-3") and cert.window == 5 and cert.key_id == signer.key_id
    assert salt == b"\x05" * 64 and cert.digest == certificate_digest(salt, reading.digest()) != reading.digest()
    assert cert.message(SESSION) == Meter("pdu-3").message(signer.key_id, SESSION, 5, cert.digest)
    assert cert.verify(key, SESSION)
    moved = [ChannelCertificate(cert.key_id, cert.channel, 6, cert.digest, cert.signature),
             ChannelCertificate(cert.key_id, Meter("pdu-4"), 5, cert.digest, cert.signature),
             ChannelCertificate(cert.key_id, cert.channel, 5, bytes(64), cert.signature),
             ChannelCertificate(cert.key_id, cert.channel, 5, cert.digest[:63], cert.signature)]
    assert not any(c.verify(key, SESSION) for c in moved) and not cert.verify(key, OTHER_SESSION)
    as_link = Certificate(cert.key_id, "pdu-3", Direction.EGRESS, 5, cert.digest, cert.signature)
    assert as_link.message(SESSION) != cert.message(SESSION) and not as_link.verify(key, SESSION)
    (tmp_path / "stranger").mkdir()
    assert not cert.verify(signers.signer(tmp_path / "stranger", seed=b"other").public_key(), SESSION)

    link = issuer.sign("lnk", Direction.EGRESS, 5, cert.digest)
    assert link is not None and link.verify(key, SESSION) and signers.used(signer) == [0, 1]
    with pytest.raises(ValueError, match="64 bytes"):
        issuer.sign_on(Meter("pdu-3"), 6, bytes(63))
    assert signers.used(signer) == [0, 1]

    signer.close()
    second = Issuer(LmsSigner(signer.key, signer.state), SESSION)
    assert certify_reading(second, Reading("pdu-3", 6, (1,)))[1] is None
    assert second.refused == [("pdu-3", 6)] and not second.started


def test_a_link_certificate_is_the_bytes_it_was(tmp_path):
    """A link's record, raw-digest and clock certificates, under fixed coins and a fresh key: their key id, link,
    direction, window, message, signed digest and signature hash to what they did when the device knew only links, and
    a link certificate is the channel certificate of its `Link`."""
    salts = iter(bytes([k]) * 64 for k in range(1, 4))
    signer = signers.signer(tmp_path)
    issuer = Issuer(signer, SESSION, coins=lambda n: next(salts))
    params = GridParams(buckets=3, allocation=2, frame_bytes=2, bucket_ms=50)
    record = Record("lnk", 7, fill([b"ab", b"cd", b"ef"], [2, 0, 1], 2), COMPLETE)
    certified = issuer.certify(record, Direction.INGRESS, params, bytes(range(64)))
    raw = issuer.sign("lnk", Direction.EGRESS, 8, bytes(range(64)))
    clock = issuer.salted_clock("lnk", 9, ClockSync(3, 1))
    certs = [certified.certificate, raw, clock.certificate]
    got = [hashlib.sha512(c.key_id + c.link.encode() + c.direction.value.encode() + c.window.to_bytes(8, "big")
                          + c.message(SESSION) + c.digest + c.signature).hexdigest() for c in certs]
    assert got == [
        "95821874c886793d46c855af683e45b452334ce2d552af38f1d1b0d550dc3d42428a48f69d0978fc8f37298b8d6f8f4ed698e0f67c81e07b"
        "ff55345d0670c175",
        "530f197f137a0ac7121d0c2569225033887a93b689662b96b2a83f4d5a81ff1cc56ab5c1597352bb3b2132556a2cbbc378c7d2c8ddedf975"
        "eb90e0e5812791e6",
        "857a9368e5ef3f862d10be1337ef64c4be3f84934b411ec94b456724f1242ad85bfa65672e85dc2f2d3559d35704775db9a4999c05768515"
        "3e2a17184b1ea1d8",
    ]
    key = signer.public_key()
    for c in certs:
        assert c.channel == Link(c.link, c.direction) and c.channel.kind == "link"
        assert c.message(SESSION) == certificate_message(c.key_id, SESSION, c.link, c.direction, c.window, c.digest)
        as_channel = ChannelCertificate(c.key_id, c.channel, c.window, c.digest, c.signature)
        assert as_channel.message(SESSION) == c.message(SESSION) and as_channel.verify(key, SESSION)
    assert channel_message(Link("lnk", Direction.EGRESS), raw.key_id, SESSION, 8, raw.digest) == raw.message(SESSION)
