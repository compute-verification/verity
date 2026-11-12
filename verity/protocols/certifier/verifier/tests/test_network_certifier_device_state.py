"""The network certifier's device state and the start's certificate: a session's base on its key's life is taken once,
written before it is returned and never given again; one run holds the state at a time; the start's certificate
binds the base, the schedule, the link, the session and the key; and its tag is no other separator's."""

import dataclasses
import json
import struct

import pytest

from verity.commitments import domains
from verity.commitments.crypto import lms
from verity.commitments.identity import tagged_sha512
from verity.protocols.certifier.verifier import certificate
from verity.protocols.certifier.verifier.certificate import (
    CertifiedStart, DeviceState, Direction, Issuer, LmsSigner, StartRefused, certificate_digest, start_digest,
    start_link)

import signers

SESSION, OTHER_SESSION = signers.SESSION, signers.OTHER_SESSION
THIRD_SESSION = bytes(range(2, 34))
SCHEDULE, OTHER_SCHEDULE = bytes(range(64)), bytes(range(1, 65))
PUBLIC = lms.public_key(signers.key())
OTHER_PUBLIC = lms.public_key(signers.key(b"another certifier"))


def state(tmp_path):
    path = tmp_path / "device.json"
    DeviceState.create(path, PUBLIC)
    return path


def on_disk(path) -> tuple[int, list[str]]:
    body = json.loads(path.read_text())
    return body["count_ms"], body["sessions"]


def test_a_session_starts_at_the_keys_life_and_moves_it_on_by_its_span(tmp_path):
    """The base is the count before the start, and the file holds the count after it before `start` returns: a run
    that ends at once has still used its whole span."""
    path = state(tmp_path)
    assert on_disk(path) == (0, [])
    with DeviceState(path, PUBLIC) as first:
        assert first.count_ms == 0 and first.start(SESSION, 1200) == 0
        assert first.count_ms == 1200 and on_disk(path) == (1200, [SESSION.hex()])
    with DeviceState(path, PUBLIC) as second:
        assert second.count_ms == 1200 and second.start(OTHER_SESSION, 600) == 1200
        assert on_disk(path) == (1800, [SESSION.hex(), OTHER_SESSION.hex()])
    assert not list(tmp_path.glob("*.tmp"))


def test_a_held_state_refuses_a_second_run_until_the_first_closes(tmp_path):
    path = state(tmp_path)
    first = DeviceState(path, PUBLIC)
    with pytest.raises(StartRefused, match="held by another run"):
        DeviceState(path, PUBLIC)
    assert first.start(SESSION, 600) == 0
    with pytest.raises(StartRefused, match="held by another run"):
        DeviceState(path, PUBLIC)
    first.close()
    first.close()
    with pytest.raises(StartRefused, match="closed"):
        first.start(OTHER_SESSION, 600)
    with DeviceState(path, PUBLIC) as second:
        assert second.start(OTHER_SESSION, 600) == 600


def test_a_session_already_started_is_refused_and_takes_no_time(tmp_path):
    path = state(tmp_path)
    with DeviceState(path, PUBLIC) as first:
        first.start(SESSION, 600)
    with DeviceState(path, PUBLIC) as again:
        with pytest.raises(StartRefused, match=f"session {SESSION.hex()} is already started"):
            again.start(SESSION, 600)
        assert again.count_ms == 600 and on_disk(path) == (600, [SESSION.hex()])
        assert again.start(OTHER_SESSION, 600) == 600
        with pytest.raises(StartRefused, match="this run has already started a session"):
            again.start(THIRD_SESSION, 600)
    assert on_disk(path) == (1200, [SESSION.hex(), OTHER_SESSION.hex()])


def test_a_state_that_is_missing_unreadable_or_another_keys_is_refused(tmp_path):
    path = tmp_path / "device.json"
    with pytest.raises(StartRefused, match="missing"):
        DeviceState(path, PUBLIC)
    with pytest.raises(StartRefused, match="no device state"):
        DeviceState(tmp_path / "absent" / "device.json", PUBLIC)
    DeviceState.create(path, PUBLIC)
    with pytest.raises(StartRefused, match="already has a device state"):
        DeviceState.create(path, PUBLIC)
    with pytest.raises(StartRefused, match="another key's"):
        DeviceState(path, OTHER_PUBLIC)
    good = path.read_text()
    body = json.loads(good)
    for bad in (good[:-2], "[]", json.dumps(body | {"count_ms": 600}), json.dumps(body | {"count_ms": -1}),
                json.dumps(body | {"sessions": [SESSION.hex()]}), json.dumps(body | {"public": OTHER_PUBLIC.hex()}),
                json.dumps({k: v for k, v in body.items() if k != "check"}), json.dumps(body | {"count_ms": "0"})):
        path.write_text(bad)
        with pytest.raises(StartRefused, match="unreadable"):
            DeviceState(path, PUBLIC)
    path.write_text(good)
    DeviceState(path, PUBLIC).close()  # each refusal above let the lock go


def test_a_state_rolled_back_under_its_run_is_refused(tmp_path):
    path = state(tmp_path)
    with DeviceState(path, PUBLIC) as first:
        first.start(SESSION, 600)
    later = path.read_text()
    with DeviceState(path, PUBLIC) as run:
        DeviceState._write(path, PUBLIC, 0, ())
        with pytest.raises(StartRefused, match="changed under this run"):
            run.start(OTHER_SESSION, 600)
        path.write_text(later)
        assert run.start(OTHER_SESSION, 600) == 600


def test_a_span_or_session_that_is_no_span_or_session_and_a_life_with_no_room(tmp_path):
    path = state(tmp_path)
    with DeviceState(path, PUBLIC) as run:
        for span in (0, -1, 1.5, True):
            with pytest.raises(ValueError, match="positive number of milliseconds"):
                run.start(SESSION, span)
        with pytest.raises(ValueError, match="32 bytes"):
            run.start(SESSION[:31], 600)
        with pytest.raises(StartRefused, match="no room"):
            run.start(SESSION, 1 << 64)
        assert on_disk(path) == (0, [])
        assert run.start(SESSION, (1 << 64) - 1) == 0
    with DeviceState(path, PUBLIC) as full:
        with pytest.raises(StartRefused, match="no room"):
            full.start(OTHER_SESSION, 1)


def test_the_start_digest_is_the_documented_bytes():
    want = tagged_sha512("verity-network-certifier/start/v1", struct.pack(">Q", 86_400_000) + SCHEDULE)
    assert start_digest(86_400_000, SCHEDULE) == want and len(want) == 64
    assert start_link("uplink") == "uplink-origin"
    assert len({start_digest(0, SCHEDULE), start_digest(1, SCHEDULE), start_digest(0, OTHER_SCHEDULE)}) == 3
    for base in (-1, 1 << 64):
        with pytest.raises(ValueError):
            start_digest(base, SCHEDULE)
    for schedule in (SCHEDULE[:63], SCHEDULE + b"\0", bytearray(SCHEDULE)):
        with pytest.raises(ValueError, match="64 bytes"):
            start_digest(0, schedule)


def test_the_starts_certificate_binds_base_schedule_link_session_and_key(tmp_path):
    signer = signers.signer(tmp_path)
    key = signer.public_key()
    issuer = Issuer(signer, SESSION)
    start = issuer.certify_start("uplink", 1200, SCHEDULE)
    c = start.certificate
    assert issuer.started and signers.used(signer) == [0]
    assert (c.link, c.direction, c.window, c.digest) == ("uplink-origin", Direction.EGRESS, 0,
                                                          start_digest(1200, SCHEDULE))
    assert start.refusal(key, SESSION, "uplink", SCHEDULE) is None and start.verify(key, SESSION, "uplink", SCHEDULE)

    def refusal(s, *, key=key, session=SESSION, link="uplink", schedule=SCHEDULE):
        return s.refusal(key, session, link, schedule)

    replace = dataclasses.replace
    assert refusal(start, link="other") == "start"
    assert refusal(replace(start, certificate=replace(c, direction=Direction.INGRESS))) == "start"
    assert refusal(replace(start, certificate=replace(c, window=1))) == "start"
    assert refusal(replace(start, certificate=replace(c, link="uplink"))) == "start"
    assert refusal(replace(start, certificate=None)) == "start"
    assert refusal(start, key=LmsSigner(signers.key(b"another certifier"), tmp_path / "none").public_key()) == "key"
    assert refusal(start, schedule=OTHER_SCHEDULE) == "schedule"
    assert refusal(replace(start, base_ms=1201)) == "opening"
    assert refusal(replace(start, base_ms=-1)) == refusal(replace(start, base_ms=1 << 64)) == "opening"
    assert refusal(replace(start, schedule=OTHER_SCHEDULE), schedule=OTHER_SCHEDULE) == "opening"
    assert refusal(replace(start, schedule=SCHEDULE[:-1]), schedule=SCHEDULE[:-1]) == "opening"
    assert refusal(start, session=OTHER_SESSION) == "signature"
    moved = replace(c, digest=start_digest(0, SCHEDULE))
    assert refusal(CertifiedStart(0, SCHEDULE, moved)) == "signature"
    # a record's or a clock's certificate signs a salted digest under another tag: none opens as a start
    record = issuer.sign("uplink-origin", Direction.EGRESS, 0,
                         certificate_digest(bytes(64), start_digest(1200, SCHEDULE)))
    assert refusal(CertifiedStart(1200, SCHEDULE, record)) == "opening"


def test_a_refused_signer_gives_no_starts_certificate(tmp_path):
    """A session its leaf state has started is refused at the start's certificate, the session's first signature."""
    first = signers.signer(tmp_path)
    assert Issuer(first, SESSION).certify_start("uplink", 0, SCHEDULE) is not None
    first.close()
    again = Issuer(LmsSigner(first.key, first.state), SESSION)
    assert again.certify_start("uplink", 600, SCHEDULE) is None
    assert again.refused == [("uplink-origin", Direction.EGRESS, 0)] and not again.started
    assert signers.used(first) == [0]


def test_the_starts_tag_is_no_other_separator_and_neither_begins_the_other():
    """The start's tag is a literal of the certificate's, as its other tags are, which the table of domain separators
    does not hold: this holds what its entry there would. It is no entry of the table and no other tag of the
    certificate's, and none of them begins it or is begun by it."""
    mine = certificate.START_TAG.encode()
    table = [v.encode() for k, v in vars(domains).items() if k.isupper() and isinstance(v, str)]
    own = [(v if isinstance(v, bytes) else v.encode()).rstrip(b"\0") for k, v in vars(certificate).items()
           if k.endswith("_TAG") and k != "START_TAG"]
    assert domains.CERTIFIER_CLOCK.encode() in table and certificate.CLOCK_TAG.encode() in own
    assert not [t for t in table + own if t.startswith(mine) or mine.startswith(t)]
