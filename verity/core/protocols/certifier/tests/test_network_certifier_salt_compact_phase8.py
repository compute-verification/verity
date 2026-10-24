"""Red-team phase 8, target 3 (CERT-VARLEN #3593 + the compact-framing branch), CPU-only.

Two questions the lead set:

  a. The salt at the head of the queue (#3593, `cursor/cert-salt-head-061a`): can the salt's position shift the
     request ends the served-counts check derives from the signed counts?
  b. Compact framing (`cursor/cert-compact-061a`): does the 8-byte (ordinal, token field) record with a client-side
     shim open any channel the 256-byte `id_event` frames did not?

Findings (each asserted below):

  a. The exactness is a premise of the proxy's placement, not a fact the record proves. `served.derive` (the body of
     `NetworkServedCounts_v1`) subtracts the window's `own` frame from bucket 0's count *unconditionally*; the only
     audit that reads the salt frame, `active.check_salts`, accepts exactly one PLACEHOLDER frame *anywhere* in the
     grid and never pins its bucket. So two records with the same real traffic but the salt in bucket 0 vs bucket 1
     derive *different* request ends, and both pass `check_salts` and `served.reference`. The honest `Bucketer`
     (`first=`, appendleft) always leaves the salt in bucket 0 slot 0, so honest runs are exact; nothing ties the
     circuit's bucket-0 assumption to the signed record.

  b. The compact record's request ordinal is pinned to the connection's stream in `_egress_record` (`got[0] != s`
     faults), so the ordinal opens no channel. The token field is free at the certifier in both schemes (the certifier
     commits whatever the server writes and the tie to the certified I/O pins it downstream), so compact does not
     widen the token channel. The one genuinely new in-band field is the response *finish*: `COMPACT_LAST` on any token
     and `COMPACT_EOS` are well-formed records the certifier commits, whereas the 256-byte `id_event` carries
     `"finish_reason":null` and marks the end out of band. The finish is thus pinned only by the downstream advice
     `finish` tie; at the certifier it is an unconstrained ~1-trit-per-response-end signal.
"""

from __future__ import annotations

import secrets
from fractions import Fraction

from verity.core.protocols.certifier import active as A
from verity.core.protocols.certifier import io_tie as IO
from verity.core.protocols.certifier import served as SV
from verity.core.protocols.certifier.grid import Certifier, GridParams, Record, COMPLETE, row_count
from verity.core.protocols.certifier.schedule import ClockSync

B = 128  # frame bytes, comfortably over SALT_FRAME_BYTES (81)
SYNC = ClockSync(0, Fraction(1, 1))
PROFILE = SV.ServedProfile(response_frames=1, request_frames=1, windows=1, own=1)


def _raw_record(grid: list) -> Record:
    return Record("ingress", 0, tuple(grid), COMPLETE)


def _salt_frame() -> bytes:
    return A.salt_frame(0, A.PLACEHOLDER, B)


def _data_frame() -> bytes:
    return A.encode_frame(A.DATA, A.CANONICAL_STREAM, b"q", B)


# --- a. the salt at the head of the queue ----------------------------------------------------------------------------

def test_honest_bucketer_places_salt_at_bucket0_slot0():
    """The honest proxy leaves the salt in bucket 0 slot 0, so the premise holds for an honest run."""
    cert = Certifier("ingress", GridParams(buckets=4, allocation=4, frame_bytes=B))
    first = lambda w: _salt_frame() if w == 0 else None
    buck = A.Bucketer(cert, first=first)
    for _ in range(2):
        buck.arrive(_data_frame(), 0)
    row, _ = buck.tick()
    kind, stream, _ = A.decode_frame(row[0])
    assert kind == A.SALT and stream == A.SALT_STREAM  # the salt is the first frame released
    assert buck.own == {0: 0}  # and the own map records it at bucket 0 slot 0


def test_derive_shifts_request_ends_when_the_salt_moves_bucket():
    """Same real traffic (one request in bucket 0, one in bucket 1); only the salt's bucket differs. `derive`
    subtracts `own` from bucket 0 either way, so a request that honestly ends at bucket 0 is derived to end at
    bucket 1 when the salt sits in bucket 1 instead. Both pass the circuit reference."""
    honest = [2, 1, 0, 0]  # salt + 1 request in bucket 0, 1 request in bucket 1
    moved = [1, 2, 0, 0]   # 1 request in bucket 0, salt + 1 request in bucket 1

    assert sum(honest) == sum(moved)  # the same number of frames on the wire
    egress_honest = SV.derive(honest, SYNC, PROFILE, allocation=4)
    egress_moved = SV.derive(moved, SYNC, PROFILE, allocation=4)
    assert egress_honest == [1, 1, 0, 0]
    assert egress_moved == [0, 2, 0, 0]
    assert egress_honest != egress_moved  # the derived request ends shifted a bucket

    # the served-counts circuit (`NetworkServedCounts_v1`) accepts each with its own self-reported egress
    assert SV.reference(4, 1, 1, 4, 4, 1, honest, egress_honest, (0, 1, 1)) is True
    assert SV.reference(4, 1, 1, 4, 4, 1, moved, egress_moved, (0, 1, 1)) is True


def test_check_salts_does_not_pin_the_salt_bucket():
    """`check_salts` is the one audit that reads the salt frame, and it accepts the salt in any bucket: it counts
    exactly one PLACEHOLDER frame anywhere in the grid. Nothing ties the circuit's bucket-0 `own` subtraction to
    the record."""
    revealed = {0: secrets.token_bytes(A.SALT_BITS // 8)}
    salt, data = _salt_frame(), _data_frame()
    in_b0 = _raw_record([(salt, data, None, None), (data, None, None, None), (None,) * 4, (None,) * 4])
    in_b1 = _raw_record([(data, None, None, None), (salt, data, None, None), (None,) * 4, (None,) * 4])

    for name, record in (("bucket 0", in_b0), ("bucket 1", in_b1)):
        audit = A.Audit()
        verdicts = A.check_salts(audit, "ingress", [record], windows=1, salts=revealed)
        assert verdicts[0].violation is None, f"check_salts rejected the salt in {name}"

    # the moved record still reports the same per-bucket counts a shifted derive would read
    assert row_count(in_b1.grid[0]) == 1 and row_count(in_b1.grid[1]) == 2


# --- b. compact framing vs the 256-byte id events --------------------------------------------------------------------

def _egress_malformed(record: bytes, stream: int) -> bool:
    """`Proxy._egress_record`'s rule (active.py: `got is None or got[0] != s`): a record that isn't one, or names
    another request than its connection's stream, is malformed."""
    got = A.compact_record_data(record)
    return got is None or got[0] != stream


def test_compact_ordinal_is_pinned_to_the_stream():
    """The request ordinal opens no channel: a record naming another request than its connection's is malformed."""
    on_stream_1 = A.compact_record(request=1, token_id=5)
    assert not _egress_malformed(on_stream_1, stream=1)       # its own request: committed
    assert _egress_malformed(on_stream_1, stream=2)           # another request: faulted
    assert A.compact_record_data(on_stream_1)[0] == 1         # the ordinal is the pinned field


def test_compact_padding_carries_no_channel():
    """Both schemes reject non-zero fill: no padding channel is opened."""
    record = A.compact_record(request=1, token_id=5)
    cell = A.compact_cell(record, B)
    assert A.compact_record_data(cell) is not None
    tampered = bytearray(cell)
    tampered[A.COMPACT_RECORD.size] = 1  # a non-zero byte past the record
    assert A.compact_record_data(bytes(tampered)) is None


def test_compact_finish_is_a_new_in_band_signal_free_at_the_certifier():
    """`COMPACT_LAST` on any token and `COMPACT_EOS` are well-formed records the certifier commits; the token field
    (finish included) is not constrained at the certifier, only pinned by the downstream advice `finish` tie. The
    256-byte `id_event` carries no in-band finish (`"finish_reason":null`), so the finish is the one new field."""
    length_on_a_token = A.compact_record(request=1, token_id=5, finish="length")
    stop = A.compact_record(request=1, token_id=None, finish="stop")
    plain = A.compact_record(request=1, token_id=5, finish=None)

    # all three are well-formed and name their stream, so the certifier commits them as given
    for record in (length_on_a_token, stop, plain):
        assert not _egress_malformed(record, stream=1)
    assert A.compact_record_data(length_on_a_token) == (1, 5, "length")
    assert A.compact_record_data(stop) == (1, None, "stop")
    assert A.compact_record_data(plain) == (1, 5, None)

    # the finish is three states (none / length / stop) the server sets per record; it is the new in-band signal
    assert set(A.COMPACT_FINISHES) == {None, "length", "stop"}

    # the 256-byte id-event scheme carries no in-band finish: `compact_record` takes a `finish` the server sets,
    # whereas an id event is (request, token_id) only and io_tie hard-codes "finish_reason":null in the frame
    import inspect

    assert "finish" in inspect.signature(A.compact_record).parameters
    assert "finish" not in inspect.signature(A.id_event).parameters
    assert b'"finish_reason":null' in IO.ID_EVENT_TAIL


def test_compact_does_not_widen_the_token_channel():
    """The token value is free at the certifier in both schemes and pinned by the tie to the certified I/O; compact
    does not add distinguishable token values the id-event scheme lacked. The one added degree of freedom is the
    finish (above)."""
    # the id-event token id is a u32 (io_tie); the compact token id is below COMPACT_LAST - 1 (31 bits) plus the
    # length flag and the EOS id, i.e. the token field is not wider than the id-event's u32 id.
    import pytest

    assert A.COMPACT_LAST == 1 << 31
    assert A.COMPACT_EOS == 0xFFFF_FFFF
    # an id event's token id spans the full u32 (io_tie.id_event raises only at 1 << 32)
    assert isinstance(IO.id_event((1 << 32) - 1), bytes)
    with pytest.raises(ValueError):
        IO.id_event(1 << 32)
    # so the token field is no wider under compact; only the finish trit is new in band
