"""`poub.wire`: the openings frame round-trips, a body it cannot read is None, and a window's frames give its
openings in order, or none at all once one frame doesn't decode (`Wire.openingsOf`)."""

from __future__ import annotations

import pytest

from verity.core.protocols.poub import audit, wire

O = audit.Opening(3, 17, bytes(range(256)) * 16, b"\1" * 1024, b"\2" * 192, (b"\3" * 64,) * 5, b"\4" * 192,
                  (b"\5" * 64,) * 9, b"")
APPENDED = audit.Opening(4, 18, bytes(range(256)) * 16, b"\1" * 1024, b"\2" * 192, (b"\3" * 64,) * 5, b"\4" * 192,
                         (b"\5" * 64,) * 3, b"", b"\6" * 64)


def test_openings_round_trip_on_the_wire_and_a_short_body_is_none():
    body = wire.encode_openings([O, APPENDED, O])
    assert wire.decode_openings(body) == [O, APPENDED, O]
    assert wire.decode_openings(body[:-1]) is None and wire.decode_openings(b"") is None
    with pytest.raises(ValueError):
        wire.encode_openings([audit.Opening(3, 17, O.block, O.leaf, O.leaf_salt, O.receipt_path, O.payload_salt,
                                            O.payload_path, b"", b"\6" * 63)])


def test_a_windows_frames_give_its_openings_in_order_or_none_once_one_frame_does_not_decode():
    frames = [wire.encode_openings([O, APPENDED]), wire.encode_openings([]), wire.encode_openings([APPENDED])]
    assert wire.openings_of(frames) == [O, APPENDED, APPENDED]
    assert wire.openings_of([]) == []
    for k in range(len(frames)):
        assert wire.openings_of(frames[:k] + [frames[k] + b"\0"] + frames[k + 1:]) == []
