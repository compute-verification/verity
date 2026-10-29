"""A8's scan of what the verifier received (`a8.scan`) cuts each drawn blob whole with the zero-knowledge drivers' one
cutter (`leak_scan.public_pieces`), leaves out the pieces the staged public circuit holds and lists them, and judges
what it finds by their one rule (`leak_scan.scan`): a chunk whose run of zero bytes the circuit file holds too reads as
no leak, a chunk piece in a verifier file, its unaligned tail included, reads as leaked, a blob shorter than a piece is
refused, and the prover's input file still holds every drawn piece."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a8  # noqa: E402

ZERO = bytes(16)


def _blobs() -> dict[str, bytes]:
    chunk = os.urandom(32) + ZERO + os.urandom(16)
    return {"chunk 7": chunk, "payload 7": os.urandom(48) + ZERO}


def _files(tmp: Path, blobs: dict[str, bytes], leak: bytes = b"") -> tuple[Path, Path, bytes]:
    """A verifier dir (the circuit, a public row, the openings it was sent, ``leak`` in them) and the prover's input file,
    each blob packed into 16-byte elements, its last one zero-padded."""
    vdir = tmp / "verifier"
    vdir.mkdir()
    public = b"g 1 2\n" + ZERO.hex().encode() + b"\n"
    (vdir / "circuit.txt").write_bytes(public)
    (vdir / "pub-1.bin").write_bytes(os.urandom(64))
    (vdir / "openings.json").write_bytes(b'[{"salt": "' + os.urandom(16).hex().encode() + b'"}]' + leak)
    control = tmp / "inst-1.bin"
    control.write_bytes(b"".join(b.ljust(-(-len(b) // 16) * 16, b"\0") for b in blobs.values()))
    return vdir, control, public


def test_a_zero_run_the_public_circuit_holds_is_left_out(tmp_path):
    blobs = _blobs()
    vdir, control, public = _files(tmp_path, blobs)
    got = a8.scan(vdir, control, blobs, set(blobs), public)
    assert got["verifier_found"] == got["leaked"] == [] and got["public_pieces"] == [ZERO.hex()]
    assert got["control"]["holds_all"] and got["control"]["blobs_found"] == got["blobs"] == 6
    assert a8.scan(vdir, control, blobs, set(blobs), b"")["verifier_found"] == ["chunk 7 @32"]


def test_a_witness_piece_the_circuit_does_not_hold_reads_as_leaked(tmp_path):
    blobs = _blobs()
    vdir, control, public = _files(tmp_path, blobs, leak=blobs["payload 7"][16:32].hex().upper().encode())
    got = a8.scan(vdir, control, blobs, set(blobs), public)
    assert got["leaked"] == ["payload 7 @16"] and got["files"]["openings.json"]["found"] == {"payload 7 @16": 1}
    assert got["placed"] == {"payload 7 @16": [("openings.json", None, "file")]}


def test_a_piece_in_a_public_row_is_excused(tmp_path):
    blobs = _blobs()
    vdir, control, public = _files(tmp_path, blobs)
    (vdir / "pub-1.bin").write_bytes(os.urandom(8) + blobs["chunk 7"][:16])
    got = a8.scan(vdir, control, blobs, set(blobs), public)
    assert got["verifier_found"] == ["chunk 7 @0"] and got["leaked"] == [] and got["structured"] == {"chunk 7 @0": "public"}


def test_a_needle_in_a_rows_unaligned_tail_is_found(tmp_path):
    """A 40-byte chunk's last 16 bytes hold no aligned piece of it, so the aligned cutter before `leak_scan.cut` never
    searched for them; written as hex into what the verifier received, they are found and leaked."""
    chunk = os.urandom(40)
    tail = chunk[-16:]
    assert all(chunk[o:o + 16] not in tail for o in range(0, 40 - 15, 16))
    vdir, control, public = _files(tmp_path, {"chunk 3": chunk}, leak=tail.hex().encode())
    got = a8.scan(vdir, control, {"chunk 3": chunk}, {"chunk 3"}, public)
    assert got["leaked"] == ["chunk 3 @24"] and got["control"]["holds_all"]


def test_a_row_shorter_than_a_piece_is_refused(tmp_path):
    """A 10-byte row has no 16-byte piece, so the aligned cutter scanned nothing of it and passed; it is refused."""
    blobs = {"chunk 1": os.urandom(32), "payload 1": os.urandom(10)}
    vdir, control, public = _files(tmp_path, blobs, leak=blobs["payload 1"])
    with pytest.raises(ValueError, match="shorter than the scan's 16-byte piece"):
        a8.scan(vdir, control, blobs, set(blobs), public)
    with pytest.raises(ValueError, match="aren't 16 bytes"):
        a8.received(vdir, control, {"payload 1": blobs["payload 1"]}, set())
