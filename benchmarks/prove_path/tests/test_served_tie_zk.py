"""`served_tie_zk.classify`: a found witness piece is ``leaked`` wherever it is, with one exemption, a piece of fewer than
DISTINCT byte values found only at fields of the proof streams' own format (offsets every repetition holds small integers
at, valued afresh in each) and not in every repetition. REDTEAM phase 16 finding 1's cases (the routed tie's switches,
low-entropy by construction, found in one repetition) are leaked."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import served_tie_zk as Z  # noqa: E402

REPS = ("s-circuit.rep0.bin", "s-circuit.rep1.bin")
SW = b"".join(x.to_bytes(2, "little") for x in (0, 1, 2, 3, 2, 1, 0, 3))
POS = (517).to_bytes(8, "little") + bytes(8)
BLOBS = {"salt @0": bytes(range(16)), "pos @0": POS, "sw_route 0 @0": SW}


def _received(found: dict[str, list[str]]) -> dict:
    files = {f: {"bytes": 1, "found": {k: 1 for k in found.get(f, [])}} for f in (*REPS, "s-session.json")}
    return {"files": files, "verifier_found": sorted({k for ks in found.values() for k in ks})}


def _streams(tmp: Path, fields: dict[int, tuple[bytes, ...]], size: int = 4096) -> Path:
    """Two repetitions' streams of random proof data, with `fields[at]`'s pieces written at offset `at` of each, and a
    session file; `a8.received`'s ``found`` read off them."""
    rnd = random.Random(5)
    for r, f in enumerate(REPS):
        b = bytearray(rnd.randbytes(size))
        for at, pieces in fields.items():
            b[at:at + 16] = pieces[r]
        (tmp / f).write_bytes(bytes(b))
    (tmp / "s-session.json").write_text("{}")
    return tmp


def _found(vdir: Path) -> dict:
    files = {f.name: {"bytes": f.stat().st_size, "found": {k: 1 for k, b in BLOBS.items() if b in f.read_bytes()}}
             for f in sorted(vdir.iterdir())}
    return {"files": files, "verifier_found": sorted({k for x in files.values() for k in x["found"]})}


def test_a_high_entropy_piece_is_leaked_anywhere():
    assert Z.classify(_received({}), BLOBS) == ([], {})
    assert Z.classify(_received({"s-session.json": ["salt @0"]}), BLOBS) == (["salt @0"], {})
    assert Z.classify(_received({REPS[0]: ["salt @0"]}), BLOBS)[0] == ["salt @0"]


def test_phase16_a_switch_leak_in_one_repetition_is_leaked():
    """REDTEAM phase 16 finding 1, flipped: the routed tie's switch row as `blobs_of` writes it is low-entropy, and found
    in one repetition it fails the statement, as in both."""
    unit = SimpleNamespace(words={"sw_route": [0, 1, 2, 3, 2, 1, 0, 3], "pos_in": list(range(300, 320))})
    blobs, _held = Z.blobs_of([unit], {})
    assert len(set(blobs["sw_route 0"])) < Z.DISTINCT <= len(set(blobs["pos_in 0"]))
    sw = "sw_route 0"
    assert Z.classify(_received({REPS[0]: [sw]}), blobs) == ([sw], {})
    assert Z.classify(_received({REPS[0]: [sw], REPS[1]: [sw]}), blobs) == ([sw], {})
    assert Z.classify(_received({REPS[0]: ["pos_in 0"]}), blobs)[0] == ["pos_in 0"]


def test_a_low_entropy_piece_is_structured_only_at_a_field_of_the_format(tmp_path):
    """At a field every repetition holds small integers at, a different value in each, a low-entropy piece found in
    one repetition is ``structured``; at an offset where the other repetition holds proof data it is leaked, as it is
    at the same field in both repetitions, beside a find in another file, or with streams of two lengths."""
    other = (2).to_bytes(8, "little") + bytes(8)
    vdir = _streams(tmp_path, {64: (POS, other), 128: (SW, other)})
    got = _found(vdir)
    assert got["verifier_found"] == ["pos @0", "sw_route 0 @0"]
    assert Z.classify(got, BLOBS, vdir) == ([], {"pos @0": 3, "sw_route 0 @0": 4})
    assert Z.placements(got, BLOBS, vdir) == {"pos @0": [(REPS[0], 64, True)], "sw_route 0 @0": [(REPS[0], 128, True)]}

    vdir = _streams(tmp_path, {64: (POS, other), 1024: (SW, (tmp_path / REPS[1]).read_bytes()[1024:1040])})
    assert Z.classify(_found(vdir), BLOBS, vdir)[0] == ["sw_route 0 @0"]

    vdir = _streams(tmp_path, {64: (POS, POS)})
    assert Z.classify(_found(vdir), BLOBS, vdir)[0] == ["pos @0"]

    vdir = _streams(tmp_path, {64: (POS, other)})
    (vdir / "s-session.json").write_bytes(POS)
    assert Z.classify(_found(vdir), BLOBS, vdir)[0] == ["pos @0"]

    vdir = _streams(tmp_path, {64: (POS, other)})
    (vdir / REPS[1]).write_bytes((vdir / REPS[1]).read_bytes() + bytes(16))
    assert Z.classify(_found(vdir), BLOBS, vdir)[0] == ["pos @0"]
