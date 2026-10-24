"""`served_tie_zk.classify`: a found witness piece is ``leaked`` wherever its find is one field's bytes (inside one field
of a proof stream's format or of a round message's) or the format isn't read, whatever its row or bytes, unless the
statement fixes its bytes (the public rows, the streams' integers, the messages' framing). A find across two fields is
no find. REDTEAM phase 16 finding 1's cases (the routed tie's switches, found in one repetition) are leaked."""
from __future__ import annotations

import json
import random
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import served_tie_zk as Z  # noqa: E402

REPS = ("s-circuit.rep0.bin", "s-circuit.rep1.bin")
SW = b"".join(x.to_bytes(2, "little") for x in (0, 1, 2, 3, 2, 1, 0, 3))
POS = b"\x5d" + bytes(15)
BLOBS = {"salt @0": bytes(range(16)), "pos @0": POS, "sw_route 0 @0": SW}
# A stream's fields as `_reads` gives them: a length, three elements, a nonce, two lengths, an element, then elements.
FIELDS = ((0, 8), (8, 16), (24, 16), (40, 16), (56, 8), (64, 8), (72, 8), (80, 16)) + tuple((96 + 16 * i, 16) for i in range(250))
NONCE = 56


def _reads(stream: bytes):
    return (FIELDS, frozenset({NONCE})) if len(stream) == 4096 else None


def _received(found: dict[str, list[str]]) -> dict:
    files = {f: {"bytes": 1, "found": {k: 1 for k in found.get(f, [])}} for f in (*REPS, "s-session.json")}
    return {"files": files, "verifier_found": sorted({k for ks in found.values() for k in ks})}


def _streams(tmp: Path, edits: dict[int, tuple[bytes, bytes]]) -> Path:
    """Two repetitions' streams of random proof data (`FIELDS`' shape), with `edits[at]` written at offset `at` of each, and
    an empty session file."""
    rnd = random.Random(5)
    for r, f in enumerate(REPS):
        b = bytearray(rnd.randbytes(4096))
        for at, xs in edits.items():
            b[at:at + len(xs[r])] = xs[r]
        (tmp / f).write_bytes(bytes(b))
    (tmp / "s-session.json").write_text("{}")
    return tmp


def _found(vdir: Path) -> dict:
    """`a8.received`'s ``found``: each piece raw or in either hex case."""
    def has(data: bytes, b: bytes) -> bool:
        return any(x in data for x in (b, b.hex().encode(), b.hex().upper().encode()))

    files = {f.name: {"bytes": f.stat().st_size, "found": {k: 1 for k, b in BLOBS.items() if has(f.read_bytes(), b)}}
             for f in sorted(vdir.iterdir())}
    return {"files": files, "verifier_found": sorted({k for x in files.values() for k in x["found"]})}


def _classify(vdir: Path):
    return Z.classify(_found(vdir), BLOBS, vdir, reads=_reads)


def test_a_find_where_the_format_isnt_read_is_leaked():
    assert Z.classify(_received({}), BLOBS) == ([], {})
    for k in BLOBS:
        assert Z.classify(_received({"s-session.json": [k]}), BLOBS) == ([k], {})
        assert Z.classify(_received({REPS[0]: [k]}), BLOBS)[0] == [k]


def test_phase16_a_switch_leak_in_one_repetition_is_leaked(tmp_path):
    """REDTEAM phase 16 finding 1, flipped: the routed tie's switch row as `blobs_of` writes it is low-entropy, and found
    in one repetition it fails the statement, as in both, and as it does written into one field of a stream, the very
    bytes a rule excusing whatever the streams hold would let through."""
    unit = SimpleNamespace(words={"sw_route": [0, 1, 2, 3, 2, 1, 0, 3], "pos_in": list(range(300, 320))})
    blobs, _held = Z.blobs_of([unit], {})
    assert len(set(blobs["sw_route 0"])) <= 4 < len(set(blobs["pos_in 0"]))
    sw = "sw_route 0"
    assert Z.classify(_received({REPS[0]: [sw]}), blobs) == ([sw], {})
    assert Z.classify(_received({REPS[0]: [sw], REPS[1]: [sw]}), blobs) == ([sw], {})
    assert Z.classify(_received({REPS[0]: ["pos_in 0"]}), blobs)[0] == ["pos_in 0"]
    vdir = _streams(tmp_path, {96: (SW, b"")})
    assert _classify(vdir) == (["sw_route 0 @0"], {})
    assert Z.placements(_found(vdir), BLOBS, vdir, reads=_reads) == {"sw_route 0 @0": [(REPS[0], 96, "field")]}


def test_a_find_across_the_formats_fields_is_no_find(tmp_path):
    """An element's last byte and the zeros of the next field (a zero element, a nonce) are two fields' bytes: a piece
    there is ``across`` in one repetition or both, at one place or two. Inside one field, or in a stream the format
    doesn't read, it is leaked."""
    zero_after = (POS[:1], POS[:1])
    vdir = _streams(tmp_path, {39: zero_after, 40: (bytes(16), bytes(16))})
    assert Z.placements(_found(vdir), BLOBS, vdir, reads=_reads) == {
        "pos @0": [(REPS[0], 39, "across"), (REPS[1], 39, "across")]}
    assert _classify(vdir) == ([], {"pos @0": "across"})

    vdir = _streams(tmp_path, {55: (POS[:1], b""), 56: (bytes(16), b"")})
    assert _classify(vdir) == ([], {"pos @0": "across"})

    vdir = _streams(tmp_path, {39: zero_after, 40: (bytes(16), bytes(16)), 112: (POS, b"")})
    assert _classify(vdir) == (["pos @0"], {})

    vdir = _streams(tmp_path, {39: zero_after, 40: (bytes(16), bytes(16))})
    (vdir / REPS[1]).write_bytes((vdir / REPS[1]).read_bytes() + bytes(16))
    assert _classify(vdir) == (["pos @0"], {})


def test_a_counted_piece_the_statement_fixes_is_public(tmp_path):
    """A piece inside a field is ``public`` when its bytes are a run of the stream's integers (not its nonces) or in the
    public rows; anywhere else it is leaked."""
    lengths = POS[:8] + bytes(8)
    vdir = _streams(tmp_path, {64: (lengths, lengths), 112: (POS, b"")})
    assert _classify(vdir) == ([], {"pos @0": "public"})

    vdir = _streams(tmp_path, {NONCE: (lengths, lengths), 112: (POS, b"")})
    assert _classify(vdir) == (["pos @0"], {})

    vdir = _streams(tmp_path, {112: (POS, b"")})
    (vdir / "pub-4.bin").write_bytes(b"\1" * 8 + POS + b"\1" * 8)
    assert _classify(vdir) == ([], {"pos @0": "public"})
    (vdir / "pub-4.bin").write_bytes(POS[:15])
    assert _classify(vdir) == (["pos @0"], {})


def test_reads_are_the_formats_fields_and_its_nonces():
    """`_reads` on a ``--zk`` stream of the format's shape, every vector empty but one zerocheck nonce, the icap hash, the
    lig.qn u64s and the zk words: its reads, in order, are the stream's every field (the root's and the tau
    commitments' empty), its nonce the zerocheck one, and a stream the format doesn't read whole has none."""
    u = lambda v: struct.pack("<Q", v)  # noqa: E731
    f, h, w = b"\xaa" * 16, b"\xbb" * 32, b"\xcc" * 32
    parts = [u(0), u(29), u(0), u(25), struct.pack("<I", 0), b"\x00", struct.pack("<I", 0)]
    parts += [u(0)] * 3 + [f] * 3 + [u(1), u(7)] + [u(0)] * 6 + [u(1), h] + [u(0)] * 9 + [u(2), u(5), u(9)] + [u(0)] * 4
    parts += [b"\x01", w, w, w, u(0), u(0), b"", f, f, u(0), b"", u(0), b"", f] + [u(0)] * 4
    stream = b"".join(parts)
    at = [sum(len(p) for p in parts[:i]) for i in range(len(parts))]
    assert Z._reads(stream) == (tuple(zip(at, map(len, parts))), frozenset({at[14]}))
    assert parts[14] == u(7)
    assert Z._reads(stream[:-8]) is None


def _framed(*records: tuple[int, int, int, bytes]) -> bytes:
    """A round message as `transcript_check.T` frames it: each record's header (op, kind, n), then its payload."""
    return b"".join(bytes([op, kind]) + bytes(6) + struct.pack("<Q", n) + body for op, kind, n, body in records)


def _transcript(tmp: Path, msgs: tuple[bytes, bytes], coin: str = "00" * 16) -> Path:
    """A session transcript of two repetitions' streams, round 1's message `msgs[K]` in repetition K's, beside streams the
    format doesn't read, holding none of the pieces."""
    for f in REPS:
        (tmp / f).write_bytes(b"\xee" * 768)
    label = _framed((2, 0, 9, b"flock-zk!" + bytes(7)))
    streams = [{"label": f"flock-circuit/fast100x2/rep{K}", "rounds": [{"msg": label.hex()}, {"msg": m.hex()}]}
               for K, m in enumerate(msgs)]
    (tmp / "s-session.json").write_text(json.dumps({"coin_tree": {"key": coin}, "streams": streams}))
    return tmp


def test_the_transcripts_round_messages_are_read_as_their_records(tmp_path):
    """A piece inside an element of a round message is leaked; across two of its records (an element's last byte and a
    zero element) it is no find; equal to a header or a label's bytes it is public; a nibble off the message's hex it is no
    place; in the rest of the transcript, or in a message that isn't framed, it is leaked."""
    el = [bytes(range(100 + 16 * i, 116 + 16 * i)) for i in range(3)]
    last = el[0][:15] + POS[:1]
    vdir = _transcript(tmp_path, (_framed((3, 2, 2, el[1] + POS)), _framed((3, 2, 2, el[1] + el[2]))))
    assert Z.placements(_found(vdir), BLOBS, vdir) == {"pos @0": [("s-session.json#rep0.r1", 32, "field")]}
    assert Z.classify(_found(vdir), BLOBS, vdir) == (["pos @0"], {})

    vdir = _transcript(tmp_path, (_framed((3, 2, 3, el[1] + last + bytes(16))), _framed((3, 1, 1, el[2]))))
    assert Z.placements(_found(vdir), BLOBS, vdir) == {"pos @0": [("s-session.json#rep0.r1", 47, "across")]}
    assert Z.classify(_found(vdir), BLOBS, vdir) == ([], {"pos @0": "across"})

    header = bytes([3, 2]) + bytes(6) + struct.pack("<Q", 2)
    blobs = {**BLOBS, "pos @0": header}
    vdir = _transcript(tmp_path, (_framed((3, 2, 2, el[1] + el[2])), _framed((4, 2, 2, b""))))
    found = {**_found(vdir), "verifier_found": ["pos @0"]}
    found["files"]["s-session.json"]["found"] = {"pos @0": 1}
    assert Z.classify(found, blobs, vdir) == ([], {"pos @0": "public"})

    m = _framed((3, 2, 2, el[1] + el[2]))
    vdir = _transcript(tmp_path, (m, m))
    text = (vdir / "s-session.json").read_text().replace(m.hex(), "0" + POS.hex() + m.hex()[33:], 1)
    (vdir / "s-session.json").write_text(text)
    assert _found(vdir)["verifier_found"] == ["pos @0"]
    assert Z.classify(_found(vdir), BLOBS, vdir) == ([], {"pos @0": "across"})

    vdir = _transcript(tmp_path, (m, m), coin=POS.hex())
    assert Z.classify(_found(vdir), BLOBS, vdir) == (["pos @0"], {})

    vdir = _transcript(tmp_path, (el[1] + POS, m))
    assert Z.placements(_found(vdir), BLOBS, vdir) == {"pos @0": [("s-session.json#rep0.r1", 16, "file")]}
    assert Z.classify(_found(vdir), BLOBS, vdir) == (["pos @0"], {})
