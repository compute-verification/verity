"""compile_slice_zk's scan of what the verifier received: every found witness piece is ``leaked`` but a low-entropy one
found only on the proof streams' own fields (`classify`), the format's integers (`integers`, off C-Flock's layout) or
a field valued afresh in each repetition; and each piece's classes are the witness fields it sets (`classes`). Nothing
here lowers or proves (`compile_slice_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import itertools
import random
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]
REPS = ("session-s-circuit.rep0.bin", "session-s-circuit.rep1.bin")


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_compile_slice_zk", HERE / "compile_slice_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Z = _load()
INT, FRESH, DATA = 64, 512, 2048
LEN = b"\x05" + bytes(15)
FRESH_PIECE = bytes(15) + b"\xc0"


def _streams(tmp: Path, edit=None) -> Path:
    """Two repetitions' streams of random proof data: at INT the same integers in both (a length 5 and a zero), at FRESH
    fifteen zeros and a byte of each repetition's own; `edit(rep, bytearray)` then changes them."""
    rnd = random.Random(7)
    for r, f in enumerate(REPS):
        b = bytearray(rnd.randbytes(4096))
        b[INT:INT + 16] = LEN
        b[FRESH:FRESH + 16] = bytes(15) + (b"\xd7", b"\xc0")[r]
        if edit:
            edit(r, b)
        (tmp / f).write_bytes(bytes(b))
    (tmp / "session-s-session.json").write_text("{}")
    return tmp


def _ints(stream: bytes) -> bytes:
    return bytes(INT) + b"\1" * 16 + bytes(len(stream) - INT - 16)


def _found(vdir: Path, blobs: dict[str, bytes]) -> dict:
    files = {f.name: {"bytes": f.stat().st_size, "found": {k: 1 for k, b in blobs.items() if b in f.read_bytes()}}
             for f in sorted(vdir.iterdir())}
    return {"files": files, "verifier_found": sorted({k for x in files.values() for k in x["found"]})}


def _classify(vdir: Path, blobs: dict[str, bytes]):
    return Z.classify(_found(vdir, blobs), blobs, vdir, ints=_ints)


def test_a_low_entropy_piece_on_the_formats_own_fields_is_structured(tmp_path):
    """The format's integers in both repetitions, and a field valued afresh found in one: structured, with their places."""
    blobs = {"wit 0 @6208": LEN, "wit 8 @2832": FRESH_PIECE}
    vdir = _streams(tmp_path)
    assert _classify(vdir, blobs) == ([], {"wit 0 @6208": [(REPS[0], INT, "int"), (REPS[1], INT, "int")],
                                           "wit 8 @2832": [(REPS[1], FRESH, "fresh")]})


def test_a_single_bit_witness_off_the_formats_fields_is_leaked(tmp_path):
    """REDTEAM's case on the compile slice: one decision bit or one hit in a zero piece, written into one repetition over
    proof data, is leaked, as it is in both, at the format's integers in one repetition only, or beside a find in
    another file; so is a piece found afresh in every repetition, or in streams of two lengths, and a high-entropy one
    anywhere."""
    bit, hit = b"\x01" + bytes(15), bytes(6) + b"\x02" + bytes(9)
    blobs = {"wit 0 @96": bit, "wit 3 @1040": hit, "wit salt 0": bytes(range(16))}

    def into(reps, at, piece):
        return lambda r, b: b.__setitem__(slice(at, at + 16), piece) if r in reps else None

    for edit in (into({0}, DATA, bit), into({0, 1}, DATA, bit), into({1}, DATA + 64, hit), into({0}, INT, bit)):
        vdir = _streams(tmp_path, edit)
        leaked, structured = _classify(vdir, blobs)
        assert leaked and not structured, edit

    vdir = _streams(tmp_path)
    (vdir / "session-s-session.json").write_bytes(LEN)
    assert _classify(vdir, {"wit 0 @6208": LEN})[0] == ["wit 0 @6208"]

    vdir = _streams(tmp_path, lambda r, b: b.__setitem__(slice(FRESH, FRESH + 16), FRESH_PIECE) if r == 0 else
                    b.__setitem__(slice(FRESH + 32, FRESH + 48), FRESH_PIECE))
    assert _classify(vdir, {"wit 8 @2832": FRESH_PIECE})[0] == ["wit 8 @2832"]

    vdir = _streams(tmp_path, lambda r, b: b.extend(bytes(16)) if r else None)
    assert _classify(vdir, {"wit 0 @6208": LEN})[0] == ["wit 0 @6208"]

    vdir = _streams(tmp_path, into({0}, DATA, bytes(range(16))))
    assert _classify(vdir, blobs)[0] == ["wit salt 0"]


def _u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def test_integers_are_the_reads_of_at_most_8_bytes_in_the_proof_format():
    """A ``--zk`` stream of the format's shape, every vector empty but the icap hashes and the u64 parameters: its
    lengths, header integers, nonces, parameters and zk tag are integers; its elements, hashes and zk words aren't."""
    f, h, w = b"\xaa" * 16, b"\xbb" * 32, b"\xcc" * 32
    parts = [(_u64(0), 1), (_u64(29) + _u64(0) + _u64(25) + struct.pack("<I", 0) + b"\x00" + struct.pack("<I", 0), 1)]
    parts += [(_u64(0), 1)] * 3 + [(f * 3, 0)] + [(_u64(0), 1)] * 5 + [(_u64(0), 1)] * 2
    parts += [(_u64(1), 1), (h, 0)]                                  # lig.icap: one hash
    parts += [(_u64(0), 1)] * 2 + [(_u64(0), 1)] * 2 + [(_u64(0), 1)] + [(_u64(0), 1)] * 2 + [(_u64(0), 1)] * 2
    parts += [(_u64(2), 1), (_u64(5) + _u64(9), 1)]                  # lig.qn: two u64s
    parts += [(_u64(0), 1)] * 4
    parts += [(b"\x01", 1), (w * 3, 0), (_u64(0), 1), (_u64(0), 1), (f * 2, 0), (_u64(0), 1), (_u64(0), 1), (f, 0)]
    parts += [(_u64(0), 1)] * 4
    stream = b"".join(p for p, _ in parts)
    want = b"".join(bytes([k]) * len(p) for p, k in parts)
    assert Z.integers(stream) == want
    assert Z.integers(stream[:-8]) == bytes(len(stream) - 8)


def test_classes_are_the_witness_fields_a_piece_sets():
    """On the demonstration class's honest unit, which takes real prefix hits, the layout tiles each row; a piece holding
    a nonzero hit sets ``wit.hit``, one holding a nonzero walk entry ``wit.cache.pfx``, and an all-zero piece none."""
    CS = Z.CS
    st, sl = CS.demo()
    S = SimpleNamespace(**st)
    c = CS.case(S, sl)
    unit = SimpleNamespace(words={r: CS.words(c.rows[r]) for r in CS.ROWS})
    for r in ("wit", "srv"):
        lay = Z.layout(S, r)
        assert [s for _c, s, _w in lay] == list(itertools.accumulate([0] + [w for *_, w in lay[:-1]]))
        assert sum(w for *_, w in lay) == CS.row_bits(S)[r]
    lay, bits = Z.layout(S, "wit"), c.rows["wit"]
    b = b"".join(x.to_bytes(2, "little") for x in unit.words["wit"])
    at = {name: next(s // 128 * 16 for c_, s, w in lay if c_ == name and any(bits[s:s + w]))
          for name in ("wit.hit", "wit.cache.pfx")}
    got = Z.classes(S, [unit], {b[o:o + 16] for o in at.values()} | {bytes(16)})
    assert got[bytes(16).hex()] == []
    for name, o in at.items():
        assert name in got[b[o:o + 16].hex()]
