"""`record`: the record entry's form, its message and its chain, and the auditor's ledger written and read through it.

The pinned digests are the auditor's ledger's bytes (`verity/challenger-auditor/v2`): a change to the form, the message's
framing or a line's text moves them."""

from __future__ import annotations

import hashlib
import json

import pytest

from verity.core.protocols import record
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.ml.commitments.identity import identity_digest_sha512

I = bytes(range(16))


def _sign(m: bytes) -> bytes:
    return b"\0\0\0\7" + hashlib.sha512(b"test-signer" + m).digest()


def _ledger() -> list[str]:
    lines: list[str] = []
    body = {"by": "lane-a", "epoch": 3, "families": [["salt", 1]], "minutes": 5, "peer": "p", "scheme": "s",
            "started_utc": "2000-01-01T00:00:00Z", "expires_utc": "2000-01-01T00:05:00Z", "note": "é"}
    lines.append(Ch.ledger_line(lines, "reserve", "s20000101T000000Z-abcd", body, I, _sign))
    lines.append(Ch.ledger_line(lines, "period-open", None, {"lane": "lane-a", "period": "p0"}, I, _sign))
    return lines


def test_the_message_frames_role_identifier_and_digest():
    d = bytes(64)
    role = Ch.LEDGER_ROLE.encode()
    assert record.message(Ch.LEDGER_ROLE, I, d) == hashlib.sha512(
        len(role).to_bytes(8, "big") + role + I + (64).to_bytes(8, "big") + d).digest()


def test_the_ledger_is_written_as_record_entries_chained_from_zero():
    lines = _ledger()
    entries = [record.of_json(json.loads(x)) for x in lines]
    assert [e["seq"] for e in entries] == [0, 1]
    assert entries[0]["prev"] == record.ZERO and entries[1]["prev"] == record.prev_of(lines[0])
    assert entries[1]["session"] is None
    assert [record.line(e) for e in entries] == lines == [Ch.ledger_text(e) for e in entries]
    assert all(x.isascii() for x in lines)
    for e in entries:
        tag = Ch.LEDGER_TAGS[e["schema"]]
        m = record.message(Ch.LEDGER_ROLE, I, identity_digest_sha512(tag, record.unsigned(e)))
        assert Ch.ledger_message(I, e) == m and e["sig"] == _sign(m).hex()


def test_the_ledgers_bytes_are_pinned():
    lines = _ledger()
    assert hashlib.sha512("".join(x + "\n" for x in lines).encode()).hexdigest()[:32] == PINNED_LEDGER
    assert Ch.ledger_message(I, json.loads(lines[1])).hex()[:32] == PINNED_MESSAGE


PINNED_LEDGER = "d5f300dcf07445940f0ed6d8f2d88916"
PINNED_MESSAGE = "abbc31ccb0c9cb297811fbaf3d79845c"


def _entry() -> dict:
    return json.loads(_ledger()[0])


@pytest.mark.parametrize("change", [
    lambda e: e.pop("sig"),
    lambda e: e.update(extra=1),
    lambda e: e.update(body=[]),
    lambda e: e.update(kind=5),
    lambda e: e.update(prev="0" * 127),
    lambda e: e.update(prev="A" * 128),
    lambda e: e.update(schema=None),
    lambda e: e.update(seq=-1),
    lambda e: e.update(seq=True),
    lambda e: e.update(seq=1.0),
    lambda e: e.update(session=5),
    lambda e: e.update(sig="abc"),
    lambda e: e.update(sig="AB"),
])
def test_of_json_refuses_an_entry_off_the_form(change):
    e = _entry()
    change(e)
    with pytest.raises(ValueError, match="a record entry is"):
        record.of_json(e)
    assert Ch._ledger_entry(Ch.ledger_text(e)) is None


def test_of_json_refuses_what_is_no_object():
    for x in (None, [], "x", 1):
        with pytest.raises(ValueError, match="a record entry is"):
            record.of_json(x)
