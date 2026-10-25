"""Red-team phase 8, target 3 (the salt at the head of the queue, and compact framing), CPU-only.

REDTEAM's seven cases (`cursor/redteam-phase8-certvarlen-061a` @d75f64a52e), with the two findings closed:

  a. `served.derive` (the body of `NetworkServedCounts_v1`) subtracts the window's `own` frames from bucket 0
     unconditionally, so the same traffic with the salt in bucket 1 derives other request ends and still passes the
     circuit. `check_salts` now takes the salt only as bucket 0's first frame, where `Bucketer` releases it, so the
     moved record is a `salt` violation.
  b. The compact record's ordinal is pinned to its connection's stream, its fill is zero and its token field is no
     wider than an id event's, but its finish is in band: `COMPACT_LAST` and `COMPACT_EOS` are records the certifier
     commits whatever they say. `compact_finish_violations` holds them to advice v0's `finish`, the rule the I/O tie
     applies.
"""

from __future__ import annotations

import inspect
import secrets
from fractions import Fraction

import pytest

from verity.core.protocols.certifier import active as A
from verity.core.protocols.certifier import served as SV
from verity.core.protocols.certifier.advice import RequestAdvice
from verity.core.protocols.certifier.grid import COMPLETE, Certifier, GridParams, Record, row_count
from verity.core.protocols.certifier.schedule import ClockSync

B = 128  # frame bytes, over the salt frame's
SYNC = ClockSync(0, Fraction(1, 1))
PROFILE = SV.ServedProfile(response_frames=1, request_frames=1, windows=1, own=1)


def _raw_record(grid: list) -> Record:
    return Record("ingress", 0, tuple(grid), COMPLETE)


def _salt_frame() -> bytes:
    return A.salt_frame(0, A.PLACEHOLDER, B)


def _data_frame() -> bytes:
    return A.encode_frame(A.DATA, A.CANONICAL_STREAM, b"q", B)


def _salt_verdict(record: Record) -> str | None:
    revealed = {0: secrets.token_bytes(A.SALT_BITS // 8)}
    return A.check_salts(A.Audit(), "ingress", [record], windows=1, salts=revealed)[0].violation


# --- a. the salt at the head of the queue ----------------------------------------------------------------------------

def test_honest_bucketer_places_salt_at_bucket0_slot0():
    """The honest proxy leaves the salt in bucket 0 slot 0, and the salt audit accepts that record."""
    cert = Certifier("ingress", GridParams(buckets=4, allocation=4, frame_bytes=B))
    buck = A.Bucketer(cert, first=lambda w: _salt_frame() if w == 0 else None)
    for _ in range(2):
        buck.arrive(_data_frame(), 0)
    row, _ = buck.tick()
    kind, stream, _ = A.decode_frame(row[0])
    assert kind == A.SALT and stream == A.SALT_STREAM
    assert buck.own == {0: 0}
    rows = [row] + [buck.tick()[0] for _ in range(3)]
    assert _salt_verdict(_raw_record(rows)) is None


def test_derive_shifts_request_ends_when_the_salt_moves_bucket():
    """Same real traffic, one request in each of buckets 0 and 1; only the salt's bucket differs. `derive` subtracts
    `own` from bucket 0 either way, so the moved record derives a request end a bucket late, and both pass the circuit
    reference. The salt audit refuses the moved one."""
    honest = [2, 1, 0, 0]
    moved = [1, 2, 0, 0]
    assert sum(honest) == sum(moved)
    egress_honest = SV.derive(honest, SYNC, PROFILE, allocation=4)
    egress_moved = SV.derive(moved, SYNC, PROFILE, allocation=4)
    assert egress_honest == [1, 1, 0, 0]
    assert egress_moved == [0, 2, 0, 0]
    assert SV.reference(4, 1, 1, 4, 4, 1, honest, egress_honest, (0, 1, 1)) is True
    assert SV.reference(4, 1, 1, 4, 4, 1, moved, egress_moved, (0, 1, 1)) is True

    salt, data = _salt_frame(), _data_frame()
    as_honest = _raw_record([(salt, data, None, None), (data, None, None, None), (None,) * 4, (None,) * 4])
    as_moved = _raw_record([(data, None, None, None), (salt, data, None, None), (None,) * 4, (None,) * 4])
    assert [row_count(r) for r in as_honest.grid] == honest
    assert [row_count(r) for r in as_moved.grid] == moved
    assert _salt_verdict(as_honest) is None
    assert _salt_verdict(as_moved) == "salt"


def test_check_salts_pins_the_salt_at_bucket0_slot0():
    """The salt audit accepts the salt only as bucket 0's first frame: in bucket 1, or behind a data frame in bucket
    0, it is a `salt` violation."""
    salt, data = _salt_frame(), _data_frame()
    empty = (None,) * 4
    at_head = _raw_record([(salt, data, None, None), (data, None, None, None), empty, empty])
    in_b1 = _raw_record([(data, None, None, None), (salt, data, None, None), empty, empty])
    behind = _raw_record([(data, salt, None, None), (data, None, None, None), empty, empty])
    alone_in_b1 = _raw_record([empty, (salt, None, None, None), empty, empty])
    assert _salt_verdict(at_head) is None
    for record in (in_b1, behind, alone_in_b1):
        assert _salt_verdict(record) == "salt"


# --- b. compact framing vs the 256-byte id events --------------------------------------------------------------------

def _egress_malformed(record: bytes, stream: int) -> bool:
    """`Proxy._egress_record`'s rule: a record that isn't one, or names another request than its connection's stream,
    is malformed."""
    got = A.compact_record_data(record)
    return got is None or got[0] != stream


def test_compact_ordinal_is_pinned_to_the_stream():
    """The request ordinal opens no channel: a record naming another request than its connection's is malformed."""
    on_stream_1 = A.compact_record(request=1, token_id=5)
    assert not _egress_malformed(on_stream_1, stream=1)
    assert _egress_malformed(on_stream_1, stream=2)
    assert A.compact_record_data(on_stream_1)[0] == 1
    with pytest.raises(ValueError):
        A.compact_record(request=0, token_id=5)


def test_compact_padding_carries_no_channel():
    """A cell with a non-zero byte past its record isn't one."""
    cell = A.compact_cell(A.compact_record(request=1, token_id=5), B)
    assert A.compact_record_data(cell) is not None
    tampered = bytearray(cell)
    tampered[A.COMPACT_RECORD.size] = 1
    assert A.compact_record_data(bytes(tampered)) is None


def test_compact_finish_is_in_band_at_the_certifier():
    """`COMPACT_LAST` on any token and `COMPACT_EOS` are well-formed records the certifier commits; an id event carries
    `"finish_reason":null` and no finish. The finish is the one field compact adds in band."""
    length_on_a_token = A.compact_record(request=1, token_id=5, finish="length")
    stop = A.compact_record(request=1, token_id=None, finish="stop")
    plain = A.compact_record(request=1, token_id=5, finish=None)
    for record in (length_on_a_token, stop, plain):
        assert not _egress_malformed(record, stream=1)
    assert A.compact_record_data(length_on_a_token) == (1, 5, "length")
    assert A.compact_record_data(stop) == (1, None, "stop")
    assert A.compact_record_data(plain) == (1, 5, None)
    assert set(A.COMPACT_FINISHES) == {None, "length", "stop"}
    assert "finish" in inspect.signature(A.compact_record).parameters
    assert "finish" not in inspect.signature(A.id_event).parameters
    assert b'"finish_reason":null' in A.ID_EVENT_TAIL


def _cells(ordinal: int, ids: list[int], finish: str) -> list[bytes]:
    *body, last = ids
    tail = A.compact_record(ordinal, None, "stop") if finish == "stop" else A.compact_record(ordinal, last, "length")
    return [A.compact_cell(A.compact_record(ordinal, t), B) for t in body] + [A.compact_cell(tail, B)]


def test_a_stop_where_the_advice_says_length_is_refused():
    """The advice's request ended at its cap; a last record claiming a stop is the one violation, at ordinal 1."""
    advice = [RequestAdvice("u", 4, 3, "length")]
    assert A.compact_finish_violations(advice, _cells(1, [7, 8, 9], "length")) == []
    assert A.compact_finish_violations(advice, _cells(1, [7, 8, 9], "stop")) == \
        [(1, "finish stop where the advice's is length")]


def test_compact_finishes_held_to_the_advice():
    """Request q is ordinal q + 1; its last record carries the advice's finish, and nothing earlier is marked."""
    advice = [RequestAdvice("a", 4, 3, "eos"), RequestAdvice("b", 2, 2, "length")]
    honest = _cells(1, [7, 8, 0], "stop") + _cells(2, [5, 6], "length")
    assert A.compact_finish_violations(advice, honest) == []
    assert A.compact_finish_violations(advice, _cells(1, [7, 8, 9], "length") + _cells(2, [5, 6], "length")) == \
        [(1, "finish length where the advice's is eos")]
    early = [A.compact_cell(A.compact_record(1, None, "stop"), B)] + _cells(1, [8, 0], "stop")
    assert A.compact_finish_violations(advice, early + _cells(2, [5, 6], "length")) == \
        [(1, "a stop finish on output 0, before the advice's last (2)")]
    short = _cells(1, [8, 0], "stop") + _cells(2, [5, 6], "length")
    assert A.compact_finish_violations(advice, short) == \
        [(1, "a stop finish on output 1, before the advice's last (2)")]
    long = _cells(1, [7, 8, 0], "stop") + [A.compact_cell(A.compact_record(1, 4), B)] + _cells(2, [5, 6], "length")
    assert A.compact_finish_violations(advice, long) == [(1, "4 records where the advice has 3 outputs")]
    unmarked = [A.compact_cell(A.compact_record(2, t), B) for t in (5, 6)]
    assert A.compact_finish_violations(advice, honest[:3] + unmarked) == \
        [(2, "finish none where the advice's is length")]
    extra = honest + [A.compact_cell(A.compact_record(3, 1), B)]
    assert A.compact_finish_violations(advice, extra) == [(3, "no request of the advice is this ordinal")]
    assert A.compact_finish_violations(advice, honest + [bytes(B - 1) + b"\x01"]) == \
        [(0, "a cell that isn't a compact record")]


def test_compact_does_not_widen_the_token_channel():
    """The compact token field is a 31-bit id with the length flag, or EOS: no wider than an id event's u32 id, which
    spans the whole u32."""
    assert A.COMPACT_LAST == 1 << 31
    assert A.COMPACT_EOS == 0xFFFF_FFFF
    assert isinstance(A.id_event(1, (1 << 32) - 1), bytes)
    with pytest.raises(ValueError):
        A.id_event(1, 1 << 32)
