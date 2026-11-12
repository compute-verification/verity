"""`poub.verifier.wire`: the openings frame round-trips, a body it cannot read is None, and a window's frames give its
openings in order, or none at all once one frame doesn't decode (`Wire.openingsOf`)."""

from __future__ import annotations

import pytest

from verity.protocols.poub.verifier import audit, wire

#: an opening of a registered block, and one of a block appended after the codec seed (its `d_j` with it)
REG = audit.Opening(3, 17, bytes(range(256)) * 16, b"\1" * 1024, b"\2" * 192, (b"\3" * 64,) * 5, b"\4" * 192,
                    (b"\5" * 64,) * 9, b"")
APPENDED = audit.Opening(4, 18, bytes(range(256)) * 16, b"\1" * 1024, b"\2" * 192, (b"\3" * 64,) * 5, b"\4" * 192,
                         (b"\5" * 64,) * 3, b"", b"\6" * 64)


def test_openings_round_trip_on_the_wire_and_a_short_body_is_none():
    body = wire.encode_openings([REG, APPENDED, REG])
    assert wire.decode_openings(body) == [REG, APPENDED, REG]
    assert wire.decode_openings(body[:-1]) is None and wire.decode_openings(b"") is None
    with pytest.raises(ValueError):
        wire.encode_openings([audit.Opening(3, 17, REG.block, REG.leaf, REG.leaf_salt, REG.receipt_path, REG.payload_salt,
                                            REG.payload_path, b"", b"\6" * 63)])


def test_a_windows_frames_give_its_openings_in_order_or_none_once_one_frame_does_not_decode():
    frames = [wire.encode_openings([REG, APPENDED]), wire.encode_openings([]), wire.encode_openings([APPENDED])]
    assert wire.openings_of(frames) == [REG, APPENDED, APPENDED]
    assert wire.openings_of([]) == []
    for k in range(len(frames)):
        assert wire.openings_of(frames[:k] + [frames[k] + b"\0"] + frames[k + 1:]) == []
