"""`leak_scan`: a found witness piece is ``leaked`` wherever its find is one field's bytes (inside one field of a proof
stream's format or one record of a round message) or the format isn't read, whatever its row or bytes, unless the
statement fixes its bytes (the public rows, the statement and circuit, the streams' integers, the messages' framing). A
find across two fields is no find. A switch row found in one repetition is leaked."""
from __future__ import annotations

import json
import random
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import leak_scan as LS  # noqa: E402

REPS = ("s-circuit.rep0.bin", "s-circuit.rep1.bin")
# The routed tie's switch row as served_tie_zk.blobs_of writes it: four values, u16 little-endian.
SW = b"".join(x.to_bytes(2, "little") for x in (0, 1, 2, 3, 2, 1, 0, 3))
POS = b"\x5d" + bytes(15)
BLOBS = {"salt @0": bytes(range(16)), "pos @0": POS, "sw_route 0 @0": SW}
# A stream's fields as `reads` gives them: a length, three elements, a nonce, two lengths, an element, then elements.
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


def _found(vdir: Path, blobs: dict[str, bytes] = BLOBS) -> dict:
    """`a8.received`'s ``found``: each piece raw or in either hex case."""
    def has(data: bytes, b: bytes) -> bool:
        return any(x in data for x in (b, b.hex().encode(), b.hex().upper().encode()))

    files = {f.name: {"bytes": f.stat().st_size, "found": {k: 1 for k, b in blobs.items() if has(f.read_bytes(), b)}}
             for f in sorted(vdir.iterdir())}
    return {"files": files, "verifier_found": sorted({k for x in files.values() for k in x["found"]})}


def _classify(vdir: Path):
    return LS.classify(_found(vdir), BLOBS, vdir, reads=_reads)


def test_a_find_where_the_format_isnt_read_is_leaked():
    assert LS.classify(_received({}), BLOBS) == ([], {})
    for k in BLOBS:
        assert LS.classify(_received({"s-session.json": [k]}), BLOBS) == ([k], {})
        assert LS.classify(_received({REPS[0]: [k]}), BLOBS)[0] == [k]


def test_phase16_a_switch_leak_in_one_repetition_is_leaked(tmp_path):
    """The routed tie's switch row is low-entropy, and found in one repetition it fails the
    statement, as in both, and as it does written into one field of a stream, the very bytes a rule excusing whatever
    the streams hold would let through."""
    sw = "sw_route 0 @0"
    assert len(set(SW)) <= 4
    assert LS.classify(_received({REPS[0]: [sw]}), BLOBS) == ([sw], {})
    assert LS.classify(_received({REPS[0]: [sw], REPS[1]: [sw]}), BLOBS) == ([sw], {})
    vdir = _streams(tmp_path, {96: (SW, b"")})
    assert _classify(vdir) == ([sw], {})
    assert LS.placements(_found(vdir), BLOBS, vdir, reads=_reads) == {sw: [(REPS[0], 96, "field")]}
    assert LS.scan(_found(vdir), BLOBS, vdir, reads=_reads) == {
        "leaked": [sw], "structured": {}, "placed": {sw: [(REPS[0], 96, "field")]}}


def test_a_find_across_the_formats_fields_is_no_find(tmp_path):
    """An element's last byte and the zeros of the next field (a zero element, a nonce) are two fields' bytes: a piece
    there is ``across`` in one repetition or both, at one place or two. Inside one field, or in a stream the format
    doesn't read, it is leaked."""
    zero_after = (POS[:1], POS[:1])
    vdir = _streams(tmp_path, {39: zero_after, 40: (bytes(16), bytes(16))})
    assert LS.placements(_found(vdir), BLOBS, vdir, reads=_reads) == {
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
    """A piece inside a field is ``public`` when its bytes are a run of the stream's integers (not its nonces), in the
    public rows, or, raw or as hex, in the statement or the circuit; anywhere else (the registration among them) it is
    leaked."""
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

    for f in LS.STATEMENT:
        (vdir / f).write_text(json.dumps({"digest": POS.hex()}))
        assert _classify(vdir) == ([], {"pos @0": "public"}), f
        (vdir / f).unlink()
    (vdir / "registration.json").write_text(json.dumps({"root": POS.hex()}))
    assert _classify(vdir) == (["pos @0"], {})


def test_a_found_piece_the_driver_cant_name_is_leaked(tmp_path):
    """`rescan` takes a run's ``received`` again with the pieces its driver recomputes: one it can't (a salt the run drew)
    is leaked, and the run's own verdict is kept as ``was``."""
    vdir = _streams(tmp_path, {39: (POS[:1], POS[:1]), 40: (bytes(16), bytes(16))})
    got = {**_found(vdir), "leaked": ["pos @0"]}
    got["verifier_found"] = sorted(got["verifier_found"] + ["wit salt 3 @0"])
    assert LS.rescan(got, BLOBS, vdir, reads=_reads) == {
        "was": ["pos @0"], "leaked": ["wit salt 3 @0"], "structured": {"pos @0": "across"},
        "placed": {"pos @0": [(REPS[0], 39, "across"), (REPS[1], 39, "across")]}}


def test_a_rescan_with_a8_searches_the_kept_files_afresh(tmp_path):
    """With `a8`, `rescan` searches the kept files for the pieces as `cut` now cuts them: one the run's cutter didn't cut
    (a row's zero-padded last element) is found, and leaked; a run's find of a row the driver recomputes that the fresh
    search doesn't repeat is dropped; a salt the run drew stays, and is leaked."""
    row = bytes(16) + b"\x07\x08\x09"
    pieces = LS.cut({"r 0": row}, 16)
    assert set(pieces) == {"r 0 @0", "r 0 @3", "r 0 @16"}
    vdir = _streams(tmp_path, {112: (pieces["r 0 @16"], b"")})
    got = {**_received({REPS[1]: ["r 0 @3", "wit salt 3 @0"]}), "leaked": ["r 0 @3", "wit salt 3 @0"]}
    a8 = SimpleNamespace(needles=lambda p: {x: k for k, x in p.items()},
                         find=lambda data, t: {k: 1 for nd, k in t.items() if nd in data})
    assert LS.rescan(got, pieces, vdir, reads=_reads) == {
        "was": ["r 0 @3", "wit salt 3 @0"], "leaked": ["r 0 @3", "wit salt 3 @0"], "structured": {},
        "placed": {"r 0 @3": [(REPS[1], None, "file")]}}
    assert LS.rescan(got, pieces, vdir, reads=_reads, a8=a8, rows={"r 0"}) == {
        "was": ["r 0 @3", "wit salt 3 @0"], "leaked": ["r 0 @16", "wit salt 3 @0"], "structured": {},
        "placed": {"r 0 @16": [(REPS[0], 112, "field")]}}


def test_reads_are_the_formats_fields_and_its_nonces():
    """`reads` on a ``--zk`` stream of the format's shape, every vector empty but one zerocheck nonce, the icap hash, the
    lig.qn grinding nonces and the zk words: its reads, in order, are the stream's every field (the root's and the tau
    commitments' empty), its nonces the zerocheck one and the two grinding nonces, and a stream the format doesn't read
    whole has none."""
    u = lambda v: struct.pack("<Q", v)  # noqa: E731
    f, h, w = b"\xaa" * 16, b"\xbb" * 32, b"\xcc" * 32
    parts = [u(0), u(29), u(0), u(25), struct.pack("<I", 0), b"\x00", struct.pack("<I", 0)]
    parts += [u(0)] * 3 + [f] * 3 + [u(1), u(7)] + [u(0)] * 6 + [u(1), h] + [u(0)] * 9 + [u(2), u(5), u(9)] + [u(0)] * 4
    parts += [b"\x01", w, w, w, u(0), u(0), b"", f, f, u(0), b"", u(0), b"", f] + [u(0)] * 4
    stream = b"".join(parts)
    at = [sum(len(p) for p in parts[:i]) for i in range(len(parts))]
    assert LS.reads(stream) == (tuple(zip(at, map(len, parts))), frozenset({at[14], at[33], at[34]}))
    assert parts[14] == u(7) and (parts[33], parts[34]) == (u(5), u(9))  # the zerocheck nonce, then lig.qn's two
    assert LS.reads(stream[:-8]) is None


def test_reads_swaps_the_reader_under_its_lock_and_restores_it(monkeypatch):
    """`reads` reads through its own recorder in place of `transcript_check.R` only while it holds ``_SWAP`` (n2_zk scans
    its members in threads), and puts the reader back even when the format refuses the stream."""
    seen, layout, R = [], LS.RD.layout, LS.RD.tc.R
    monkeypatch.setattr(LS.RD, "layout", lambda b, zk: seen.append((LS._SWAP.locked(), LS.RD.tc.R)) or layout(b, zk=zk))
    assert LS.reads.__wrapped__(b"\x01") is None
    assert len(seen) == 1 and seen[0][0] and issubclass(seen[0][1], R) and seen[0][1] is not R
    assert LS.RD.tc.R is R and not LS._SWAP.locked()


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
    assert LS.placements(_found(vdir), BLOBS, vdir) == {"pos @0": [("s-session.json#rep0.r1", 32, "field")]}
    assert LS.classify(_found(vdir), BLOBS, vdir) == (["pos @0"], {})

    vdir = _transcript(tmp_path, (_framed((3, 2, 3, el[1] + last + bytes(16))), _framed((3, 1, 1, el[2]))))
    assert LS.placements(_found(vdir), BLOBS, vdir) == {"pos @0": [("s-session.json#rep0.r1", 47, "across")]}
    assert LS.classify(_found(vdir), BLOBS, vdir) == ([], {"pos @0": "across"})

    header = bytes([3, 2]) + bytes(6) + struct.pack("<Q", 2)
    blobs = {**BLOBS, "pos @0": header}
    vdir = _transcript(tmp_path, (_framed((3, 2, 2, el[1] + el[2])), _framed((4, 2, 2, b""))))
    found = {**_found(vdir), "verifier_found": ["pos @0"]}
    found["files"]["s-session.json"]["found"] = {"pos @0": 1}
    assert LS.classify(found, blobs, vdir) == ([], {"pos @0": "public"})

    m = _framed((3, 2, 2, el[1] + el[2]))
    vdir = _transcript(tmp_path, (m, m))
    text = (vdir / "s-session.json").read_text().replace(m.hex(), "0" + POS.hex() + m.hex()[33:], 1)
    (vdir / "s-session.json").write_text(text)
    assert _found(vdir)["verifier_found"] == ["pos @0"]
    assert LS.classify(_found(vdir), BLOBS, vdir) == ([], {"pos @0": "across"})

    vdir = _transcript(tmp_path, (m, m), coin=POS.hex())
    assert LS.classify(_found(vdir), BLOBS, vdir) == (["pos @0"], {})

    vdir = _transcript(tmp_path, (el[1] + POS, m))
    assert LS.placements(_found(vdir), BLOBS, vdir) == {"pos @0": [("s-session.json#rep0.r1", 16, "file")]}
    assert LS.classify(_found(vdir), BLOBS, vdir) == (["pos @0"], {})


def _zk_stream(qn: int = 2, fold: int = 2, cbn: int = 2, consn: int = 2, rs: tuple[tuple[int, int], ...] = ()) -> bytes:
    """A minimal valid ``--zk`` proof stream (Sha256 Merkle kind, unsalted) in `redigest.layout`'s read order, with the
    Ligerito grinding-nonce vectors ``lig.qn``, ``lig.fold``, ``lig.cbn`` and ``lig.consn`` populated, and the ring
    switches `rs`, each (its elements, its nonce)."""
    u, u32 = lambda v: struct.pack("<Q", v), lambda v: struct.pack("<I", v)  # noqa: E731
    f = b"\xa0" * 16
    p = [u(0), u(0), u(0), u(0), u32(0), b"\x00", u32(0)]          # cap vec 0; four ints; t=0; k=0 (Sha256)
    p += [u(0), u(0), u(0), f, f, f, u(0)]                         # zc: ab/c/rounds empty; fa/fb/fc; nonce vec 0
    p += [u(0), u(0), u(0), u(0)]                                  # lc: rounds/zp/me empty; nonce vec 0
    p += [u(len(rs))] + [x for n, nonce in rs for x in (u(n), *[f] * n, u(nonce))]  # ring switches: s_hat_v, nonce
    p += [u(0), u(0)]                                              # batching.nonce vec 0; lig.icap vec 0
    p += [u(0), u(0), u(0), u(0), u(0), u(0), u(0), u(0), u(0)]    # l0 (2); rcaps; l{i}; yr; final (2); sc; msg
    p += [u(qn)] + [u(0)] * qn + [u(0)] + [u(fold)] + [u(0)] * fold  # qn; ood vec 0; fold
    p += [u(cbn)] + [u(0)] * cbn + [u(consn)] + [u(0)] * consn  # cbn; consn
    p += [b"\x01", b"\xe0" * 32, b"\xe0" * 32, b"\xe0" * 32, u(0)]  # zk.tag; e0/e1/tp; ybar vec 0
    p += [u(0), f, f, u(0), u(0), f, u(0), u(0), u(0), u(0)]        # in.root 0; rho; sigma; tauc 0; taus 0; tau; y/open/salt/path 0
    return b"".join(p)


def test_the_ligerito_grinding_nonces_are_not_reader_fixed(tmp_path):
    """``lig.qn``, ``lig.fold``, ``lig.cbn`` and ``lig.consn`` are grinding nonces (`transcript_check`'s ``t.pow``;
    `redigest`'s ``lig-query-nonce`` and ``lig-claim-nonce`` flip them and the verifier accepts), so the prover controls
    them and `public_bytes` must not take them for the reader-fixed integers. The switch row leaked in one field, with
    its bytes also written across two grinding nonces as a ``public`` alibi, is still leaked (which a nonce-blind scan lets
    through)."""
    stream = _zk_stream()
    off = LS.RD.layout(stream, zk=True)
    _fields, nonces = LS.reads(stream)
    pieces = {"sw_route 0 @0": SW}
    (tmp_path / "s-session.json").write_text("{}")
    for k in ("lig.qn", "lig.fold", "lig.cbn", "lig.consn"):
        assert off[k] and set(off[k]) <= nonces, f"{k} is not counted as a nonce"
        b = bytearray(stream)
        b[off["zk.e0"][0]:off["zk.e0"][0] + 16] = SW  # a genuine in-one-field leak (a hiding mask the prover controls)
        b[off[k][0]:off[k][0] + 16] = SW  # the same bytes across two of k's grinding nonces (the alibi)
        (tmp_path / REPS[0]).write_bytes(bytes(b))
        assert LS.classify(_found(tmp_path, pieces), pieces, tmp_path) == (["sw_route 0 @0"], {}), k


def test_a_zero_ring_switch_nonce_is_reader_fixed(tmp_path):
    """The verifier requires each ring switch's grinding nonce to be 0 (Flock's PROTOCOL.md §8.3), so a zero one is a
    field the reader fixes: a piece found inside a field whose bytes are a ring switch's nonce and the next one's length
    is ``public``, like the stream's other integers. A nonzero one, which the verifier refuses, stays the prover's, and
    the same find is leaked."""
    k = "srv 0 @0"
    (tmp_path / "s-session.json").write_text("{}")
    for nonce, want in ((0, ([], {k: "public"})), (5, ([k], {}))):
        stream = _zk_stream(rs=((1, nonce), (3, 0)))
        off = LS.RD.layout(stream, zk=True)
        at, e0 = off["rs.0.nonce"][0], off["zk.e0"][0]
        piece = stream[at:at + 16]
        assert piece == struct.pack("<QQ", nonce, 3)
        assert (at in LS.reads(stream)[1]) == (nonce != 0) and off["rs.1.nonce"][0] not in LS.reads(stream)[1]
        b = bytearray(stream)
        b[e0:e0 + 16] = piece  # counted: inside one field, zk.e0
        (tmp_path / REPS[0]).write_bytes(bytes(b))
        assert LS.classify(_found(tmp_path, {k: piece}), {k: piece}, tmp_path) == want, nonce


#: `a8`'s piece size and finder, as `public_pieces` calls them: each piece raw or as hex, found anywhere in the data.
A8 = SimpleNamespace(PIECE=16, needles=lambda pieces: {nd: k for k, x in pieces.items() for nd in (x, x.hex().encode())},
                     find=lambda data, table: {k: 1 for nd, k in table.items() if nd in data})


def test_a_rows_tail_is_cut_and_a_leak_written_only_there_is_found(tmp_path):
    """`cut` covers each row whole: its aligned 16-byte pieces, then, when its length isn't a multiple of 16, its last 16
    bytes over the piece before them (the corpus's ``wit`` tail is 12 bytes, the demo's ``srv`` tail 2). A row whose
    witness is only in its tail, its aligned bytes zeros the circuit holds, so has a piece to scan for, and that piece
    written into one field of a stream, or the row into a file whose format isn't read, is leaked. A row shorter than a
    piece is refused: `a8.needles` wouldn't cut it."""
    for n in (16, 18, 28, 32, 196):
        row = bytes(range(n))
        pieces = LS.cut({"r": row}, 16)
        at = sorted(int(x.rsplit("@", 1)[1]) for x in pieces)
        assert at == [*range(0, n - 15, 16)] + [n - 16, n - n % 16] * (n % 16 > 0), n
        assert all(row[o:o + 16].ljust(16, b"\0") == pieces[f"r @{o}"] for o in at)
        assert {i for o in at for i in range(o, min(o + 16, n))} == set(range(n))
    wit = bytes(16) + b"\x07\x00\x09\x00"
    kept, held, public = LS.public_pieces(A8, {"wit 0": wit}, {"wit 0"}, b"g 1 2\n" + bytes(16).hex().encode())
    assert (kept, held, public) == ({"wit 0 @4": wit[4:], "wit 0 @16": wit[16:] + bytes(12)}, {"wit 0 @4", "wit 0 @16"},
                                    [bytes(16).hex()])
    assert LS.public_pieces(A8, {"wit 0": wit}, {"wit 0"}, bytes(16).hex().encode(), fields=True)[1] == {"wit 0 @4"}
    vdir = _streams(tmp_path, {112: (wit[4:], b"")})
    assert LS.classify(_found(vdir, kept), kept, vdir, reads=_reads) == (["wit 0 @4"], {})
    (tmp_path / "file").mkdir()
    vdir = _streams(tmp_path / "file", {})
    (vdir / "openings.json").write_bytes(wit)
    assert LS.classify(_found(vdir, kept), kept, vdir, reads=_reads) == (["wit 0 @4"], {})
    with pytest.raises(ValueError, match="shorter than the scan's 16-byte piece.*find_whole"):
        LS.cut({"cin 0": bytes(12)}, 16)


def test_a_short_row_is_found_whole_and_judged_by_the_rule(tmp_path):
    """`find_whole` finds a row shorter than a piece (a 12-byte carry row, an 8-byte served row) whole, each distinct row
    once: planted inside one field of a stream it is leaked; across two fields it is no find; with its bytes in the public
    rows it is public; as hex in a file the format doesn't read it is leaked, in either case, but a nibble off that hex it
    isn't found. The prover's control is to hold every row, the files searched are the piece scan's, and a row of 16
    bytes or more, or under 6, is refused."""
    row, srb = bytes(range(0xa0, 0xac)), struct.pack("<4H", 1, 0, 0xa5, 3)
    rows = {"cout 0": row, "cin 1": row, "srb 0": srb}
    (tmp_path / "v").mkdir()
    vdir = _streams(tmp_path / "v", {112: (row, b"")})
    got = LS.find_whole(None, rows, vdir, reads=_reads)
    assert (got["rows"], got["verifier_found"], got["files"][REPS[0]]["found"]) == (2, ["cout 0"], {"cout 0": 1})
    assert {k: got[k] for k in ("leaked", "structured", "placed")} == {
        "leaked": ["cout 0"], "structured": {}, "placed": {"cout 0": [(REPS[0], 112, "field")]}}
    assert LS.find_whole({"files": {REPS[1]: {}}}, rows, vdir, reads=_reads)["verifier_found"] == []

    vdir = _streams(tmp_path / "v", {90: (row, b"")})
    got = LS.find_whole(None, rows, vdir, reads=_reads)
    assert (got["leaked"], got["structured"]) == ([], {"cout 0": "across"})

    vdir = _streams(tmp_path / "v", {112: (row, b"")})
    (vdir / "pub-4.bin").write_bytes(b"\1" * 8 + row)
    assert LS.find_whole(None, rows, vdir, reads=_reads)["structured"] == {"cout 0": "public"}
    (vdir / "pub-4.bin").unlink()

    for hx, want in ((srb.hex(), ["srb 0"]), (srb.hex().upper(), ["srb 0"]), ("0" + srb.hex(), []), ("0" + srb.hex() + "0", [])):
        (vdir / "openings.json").write_text(json.dumps([{"salt/o0": hx}]))
        got = LS.find_whole(None, rows, vdir, reads=_reads)
        assert (got["verifier_found"], got["leaked"]) == (["cout 0", *want], ["cout 0", *want]), hx
    assert got["placed"] == {"cout 0": [(REPS[0], 112, "field")]}
    (vdir / "openings.json").write_text(json.dumps([{"salt/o0": srb.hex()}]))
    assert LS.find_whole(None, rows, vdir, reads=_reads)["placed"]["srb 0"] == [("openings.json", None, "file")]

    (tmp_path / "inst-2.bin").write_bytes(bytes(5) + row + bytes(3) + srb)
    assert LS.find_whole(None, rows, vdir, tmp_path / "inst-2.bin", reads=_reads)["control"] == {
        "file": "inst-2.bin", "holds_all": True, "missing": []}
    (tmp_path / "inst-2.bin").write_bytes(bytes(5) + row)
    assert LS.find_whole(None, rows, vdir, tmp_path / "inst-2.bin", reads=_reads)["control"]["missing"] == ["srb 0"]
    assert LS.find_whole(None, rows, vdir, tmp_path / "inst-2.bin", {"cin 1"}, reads=_reads)["control"]["holds_all"]
    for n in (5, 16):
        with pytest.raises(ValueError, match="aren't 6 to 15 bytes"):
            LS.find_whole(None, {**rows, "x": bytes(n)}, vdir, reads=_reads)


@pytest.mark.parametrize("t", range(1, 16))
def test_a_row_packed_whole_from_an_element_boundary_is_leaked_by_its_last_element(tmp_path, t):
    """A row of 16 zeros the circuit holds and then t witness bytes, packed into 16-byte
    elements from an element boundary (a stream's elements, each its own read; a round message's, each its own record),
    holds its tail in its last element, zero-padded. The overlapping tail piece straddles that element and the one
    before it (``across``), so only the zero-padded last chunk is one field's bytes."""
    row = bytes(16) + bytes(range(1, t + 1))
    kept = LS.public_pieces(A8, {"wit 0": row}, {"wit 0"}, b"g 1 2\n" + bytes(16).hex().encode())[0]
    vdir = _streams(tmp_path, {112: (row + bytes(16 - t), b"")})
    assert LS.classify(_found(vdir, kept), kept, vdir, reads=_reads)[0] == ["wit 0 @16"]
    (tmp_path / "msg").mkdir()
    vdir = _transcript(tmp_path / "msg", (_framed((3, 2, 2, row + bytes(16 - t))), _framed((3, 1, 1, bytes(range(100, 116))))))
    assert LS.classify(_found(vdir, kept), kept, vdir)[0] == ["wit 0 @16"]
