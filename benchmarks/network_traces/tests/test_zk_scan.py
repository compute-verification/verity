"""zk_scan's rule: a witness piece found in what the verifier received is excluded only when its value occurs in the bytes
the statement fixes (the circuit, the public rows, the transcript's framing, the proof container's fields), and a hex
match counts only nibble-aligned. Nothing here proves; the transcript is built with the spec transcript's own encoders
(`transcript_check.T`)."""

from __future__ import annotations

import importlib.util
import json
import random
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Z = _load("network_traces_zk_scan", HERE / "zk_scan.py")
sys.path.insert(0, str(REPO / "verity/ml/flock/verifier"))
import transcript_check as TC  # noqa: E402

LABEL = b"verity-flock-zk-pads-v0"
#: the label's nine padding bytes, then an observed-bytes header of 64: REDTEAM phase 18's ``wit 8 @1152``
ON_FRAMING = bytes(9) + b"\x05" + bytes(6)
SPARSE = bytes(6) + b"\x02" + bytes(9)


def _msg(elements=(), observed=b"\xaa" * 64) -> bytes:
    """One round message: the domain, a label, one element, a slice, observed bytes, a nonce and a squeeze."""
    t = TC.T([], b"flock-circuit/fast100x2/rep0")
    t.label(LABEL)
    t.obs_bytes(observed)
    t.obs(0x1234)
    t.obs_slice(list(elements) or [7, 8])
    t.pow(0)
    return t.pending + TC.T.hdr(4, 2, 6)


def _u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def _stream(payload: bytes = b"\xaa" * 16) -> tuple[bytes, bytes]:
    """A ``--zk`` stream of C-Flock's container (`redigest.layout`), every vector empty but one ring switch of one
    element, the icap's one hash and two query nonces, and its format mask; `payload` is the ring switch's element."""
    f, h, w = b"\xaa" * 16, b"\xbb" * 32, b"\xcc" * 32
    parts = [(_u64(0), 1), (_u64(29) + _u64(0) + _u64(25) + struct.pack("<I", 0) + b"\x00" + struct.pack("<I", 0), 1)]
    parts += [(_u64(0), 1)] * 3 + [(f * 3, 0)] + [(_u64(0), 1)] * 5
    parts += [(_u64(1), 1), (_u64(1), 1), (payload, 0), (_u64(0), 1), (_u64(0), 1)]     # one ring switch, its nonce 0
    parts += [(_u64(1), 1), (h, 0)]                                                     # lig.icap: one hash
    parts += [(_u64(0), 1)] * 9
    parts += [(_u64(2), 1), (_u64(0) + _u64(9), 0)]                                     # lig.qn: two nonces, messages
    parts += [(_u64(0), 1)] * 4
    parts += [(b"\x01", 0), (w * 3, 0), (_u64(0), 1), (_u64(0), 1), (f * 2, 0), (_u64(0), 1), (_u64(0), 1), (f, 0)]
    parts += [(_u64(0), 1)] * 4
    return b"".join(p for p, _ in parts), b"".join(bytes([k]) * len(p) for p, k in parts)


def test_framing_is_the_records_headers_strings_and_padding():
    """Each record's header, the domain's and label's bytes and every padding are framing; the element, the slice, the
    observed bytes and the nonce are messages, so each cuts the run. A record that doesn't parse ends it."""
    m = _msg()
    hdr, pad = TC.T.hdr, TC.T.pad
    want = [hdr(1, 0, 28) + pad(b"flock-circuit/fast100x2/rep0") + hdr(2, 0, len(LABEL)) + pad(LABEL) + hdr(5, 0, 64),
            hdr(3, 1, 1), hdr(3, 2, 2), hdr(5, 0, 8), bytes(8) + hdr(4, 2, 6)]
    assert Z.framing(m) == want
    assert sum(map(len, want)) + 64 + 16 + 32 + 8 == len(m)
    assert ON_FRAMING in want[0]
    bad = bytearray(m)
    bad[m.index(hdr(3, 2, 2))] = 9
    assert Z.framing(bytes(bad)) == want[:2]


def test_format_runs_are_the_containers_integers_and_the_ring_switch_nonce():
    """The container's lengths and parameters, and each ring switch's grinding nonce (§12.2's O4 fixes it at 0), are
    format; its elements, hashes, zk words and query nonces aren't; nothing is when the layout doesn't parse."""
    stream, mask = _stream()
    assert Z.format_mask(stream) == mask
    assert b"".join(Z.format_runs(stream)) == bytes(b for b, k in zip(stream, mask) if k)
    assert Z.format_mask(stream[:-8]) == bytes(len(stream) - 8)


def _vdir(tmp: Path, msg: bytes, stream: bytes, pub: bytes = b"") -> Path:
    rec = {"streams": [{"stream": "circuit/rep0", "rounds": [{"msg": msg.hex(), "n": 6}]}]}
    (tmp / "session-s-session.json").write_text(json.dumps(rec))
    (tmp / "session-s-circuit.rep0.bin").write_bytes(stream)
    (tmp / "pub-1.bin").write_bytes(pub or random.Random(3).randbytes(64))
    return tmp


def test_a_found_piece_is_excluded_only_by_a_value_the_statement_fixes(tmp_path):
    """REDTEAM phase 18: a piece on the label's padding and the next header is excluded (framing), as is one the public
    rows or the circuit hold; a sparse one-byte witness in a proof element, or in a transcript message, is leaked
    whatever its byte count, and so is a salt."""
    salt = bytes(range(16))
    sparse_msg = _msg(elements=[int.from_bytes(SPARSE, "little"), 8])
    blobs = {"wit 8": ON_FRAMING, "wit 3": SPARSE, "wit salt 0": salt}
    rc = Z.received(_vdir(tmp_path, sparse_msg, _stream(salt)[0]), None, blobs, set())
    assert rc["verifier_found"] == ["wit 3 @0", "wit 8 @0", "wit salt 0 @0"]
    assert rc["excluded"] == {"wit 8 @0": {"hex": ON_FRAMING.hex(), "public": ["framing"]}}
    assert rc["leaked"] == ["wit 3 @0", "wit salt 0 @0"]
    assert rc["files"]["session-s-session.json"]["found"] == ["wit 3 @0", "wit 8 @0"]

    d = tmp_path / "pub"
    d.mkdir()
    rc = Z.received(_vdir(d, _msg(observed=salt * 4), _stream(SPARSE)[0], pub=b"\x11" * 8 + SPARSE), None, blobs, set(),
                    circuit=f"g 1 2\nc {salt.hex()}\n".encode())
    assert rc["excluded"] == {"wit 3 @0": {"hex": SPARSE.hex(), "public": ["pub"]},
                              "wit 8 @0": {"hex": ON_FRAMING.hex(), "public": ["framing"]},
                              "wit salt 0 @0": {"hex": salt.hex(), "public": ["circuit"]}}
    assert rc["leaked"] == []
    assert Z.received(d, None, blobs, set(), public=[SPARSE])["excluded"]["wit 3 @0"]["public"] == ["extra", "pub"]


def test_the_ring_switch_nonce_is_format_but_a_value_across_it_and_an_element_is_not(tmp_path):
    """REDTEAM's ``wit 8 @5088``: a nonce's last byte and the two lengths after it lie on format; a piece reaching one
    byte into the element before the nonce does not."""
    stream, _mask = _stream(b"\xaa" * 15 + b"\x03")
    [at] = Z._layout().layout(stream, zk=True)["rs.0.nonce"]
    on, across = stream[at + 1:at + 17], stream[at - 1:at + 15]
    assert (on, across) == (bytes(15) + b"\x01", b"\x03" + bytes(15))
    rc = Z.received(_vdir(tmp_path, _msg(), stream), None, {"wit 8": on, "wit 9": across}, set())
    assert rc["verifier_found"] == ["wit 8 @0", "wit 9 @0"]
    assert list(rc["excluded"]) == ["wit 8 @0"] and rc["excluded"]["wit 8 @0"]["public"] == ["format"]
    assert rc["leaked"] == ["wit 9 @0"]


def test_hex_counts_only_nibble_aligned(tmp_path):
    """A piece's hex found at an odd offset into a run of hex digits straddles nibbles and isn't found; at an even one it
    is, in either case."""
    x = bytes(10) + b"\x20" + bytes(5)
    shifted = "0" + x.hex() + "0"
    table = Z.needles({"wit 8 @4176": x})
    assert x.hex() in shifted and Z.find(shifted.encode(), table) == set()
    assert Z.find(('"' + x.hex().upper() + '"').encode(), table) == {"wit 8 @4176"}
    assert Z.find(("00" + x.hex()).encode(), table) == {"wit 8 @4176"}
    assert Z.find(b"\x07" + x, table) == {"wit 8 @4176"}


def test_pieces_carry_rows_whole_tails_and_the_control(tmp_path):
    """A blob under 16 bytes (a 12-byte carry row) is one piece; a row of 20 bytes is cut at 0 and at its last 16; each
    value once. The control is to hold every held piece."""
    row = bytes(range(1, 21))
    ps, held = Z.pieces({"cin 6": bytes(12), "wit 0": row, "wit 1": row}, {"wit 0", "wit 1"})
    assert ps == {"cin 6 @0": bytes(12), "wit 0 @0": row[:16], "wit 0 @4": row[4:]}
    assert held == {"wit 0 @0", "wit 0 @4"} and Z.offsets(32) == [0, 16] and Z.offsets(12) == [0]
    vdir = tmp_path / "v"
    vdir.mkdir()
    (vdir / "proof.bin").write_bytes(b"\x01" * 40)
    ctl = tmp_path / "inst.bin"
    ctl.write_bytes(b"\x09" + row + bytes(12))
    rc = Z.received(vdir, ctl, {"cin 6": bytes(12), "wit 0": row}, {"wit 0", "cin 6"})
    assert rc["control"]["holds_all"] and rc["verifier_found"] == [] and rc["pieces"] == 3
    ctl.write_bytes(row[:16])
    rc = Z.received(vdir, ctl, {"wit 0": row}, {"wit 0"})
    assert not rc["control"]["holds_all"] and rc["control"]["missing"] == ["wit 0 @4"]
