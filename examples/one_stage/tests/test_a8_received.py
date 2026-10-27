"""A8's scan of what the verifier received (`a8.received`) leaves out the pieces the staged public circuit holds and lists
them: a block whose run of zero bytes the circuit file holds too reads as no leak, while a block piece in a verifier file
still does, and the instance file still holds every drawn blob."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a8  # noqa: E402

ZERO = bytes(16)


def _blobs() -> dict[str, bytes]:
    block = os.urandom(32) + ZERO + os.urandom(16)
    return {"block 7": block, "payload 7": os.urandom(48) + ZERO}


def _files(tmp: Path, blobs: dict[str, bytes], leak: bytes = b"") -> tuple[Path, Path, bytes]:
    vdir = tmp / "verifier"
    vdir.mkdir()
    public = b"g 1 2\n" + ZERO.hex().encode() + b"\n"
    (vdir / "circuit.txt").write_bytes(public)
    (vdir / "pub-1.bin").write_bytes(os.urandom(64) + leak)
    control = tmp / "inst-1.bin"
    control.write_bytes(b"".join(blobs.values()))
    return vdir, control, public


def test_a_zero_run_the_public_circuit_holds_is_left_out(tmp_path):
    blobs = _blobs()
    vdir, control, public = _files(tmp_path, blobs)
    got = a8.received(vdir, control, blobs, set(blobs), public)
    assert got["verifier_found"] == [] and got["public_pieces"] == [ZERO.hex()]
    assert got["control"]["holds_all"] and got["control"]["blobs_found"] == 2
    assert a8.received(vdir, control, blobs, set(blobs))["verifier_found"] == ["block 7"]


def test_a_witness_piece_the_circuit_does_not_hold_still_reads_as_a_leak(tmp_path):
    blobs = _blobs()
    vdir, control, public = _files(tmp_path, blobs, leak=blobs["payload 7"][16:32].hex().upper().encode())
    got = a8.received(vdir, control, blobs, set(blobs), public)
    assert got["verifier_found"] == ["payload 7"] and got["files"]["pub-1.bin"]["found"] == {"payload 7": 1}


def test_a_blob_with_no_piece_left_is_not_asked_of_the_control(tmp_path):
    blobs = {"block 0": ZERO * 2, "block 1": os.urandom(32)}
    kept, out = a8.public_pieces(blobs, ZERO)
    assert kept == {"block 0": b"", "block 1": blobs["block 1"]} and out == [ZERO.hex()]
    vdir, control, public = _files(tmp_path, blobs)
    got = a8.received(vdir, control, blobs, set(blobs), public)
    assert got["control"]["holds_all"] and got["control"]["missing"] == []
