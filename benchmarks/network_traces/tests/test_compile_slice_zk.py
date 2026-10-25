"""compile_slice_zk proves w212 at its run's engine class, scheduler included; its scans of what the verifier received,
of the witness rows and salts and of the carry rows, are `zk_scan`'s (tests/test_zk_scan.py), and each found witness
piece's classes are the witness fields it sets (`classes`, `scanned`). Nothing here lowers or proves
(`compile_slice_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_compile_slice_zk", HERE / "compile_slice_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Z = _load()


def test_w212_is_proved_at_its_runs_engine_class(monkeypatch):
    """8eb1's engine class is its trace header's, AsyncScheduler (AS = 1) included, and w212 is loaded at it. A fixture
    at another scheduler, the sync class's (AS = 0), is refused, so every window of the run is proved at its one AS."""
    x = Z.load("w212")
    assert Z.engine_class()["async_scheduling"] is True
    assert x.fx["engine"] == Z.engine_class() and x.st["AS"] == 1 and x.sl.steps == 21
    monkeypatch.setattr(Z, "W212", Z.FIXTURES / "compile_slice_w212_class.json")
    with pytest.raises(SystemExit, match="isn't the run's engine class"):
        Z.load("w212")


def test_classes_are_the_witness_fields_a_piece_sets():
    """On the demonstration class's honest unit, which takes real prefix hits, the layout tiles each row; a piece holding
    a nonzero hit sets ``wit.hit``, one holding a nonzero walk entry ``wit.cache.pfx``, and an all-zero piece none."""
    CS = Z.CS
    st, sl = CS.demo()
    S = SimpleNamespace(**st)
    c = CS.case(S, sl, Z.BOOK)
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


def test_scanned_gives_each_found_piece_its_classes():
    """A found piece, excluded or leaked, is named with the witness classes it sets; a salt sets none."""
    CS = Z.CS
    st, sl = CS.demo()
    S = SimpleNamespace(**st)
    c = CS.case(S, sl, Z.BOOK)
    unit = SimpleNamespace(words={r: CS.words(c.rows[r]) for r in CS.ROWS}, chained=True)
    salts = {r: [bytes(range(16))] for r in ("adv", "cert", "srv", "wit", "cin", "cout")}
    blobs, _held = Z.blobs_of([unit], salts, SimpleNamespace(public=False))
    ps = Z.ZS.pieces(blobs)[0]
    hit = next(k for k, x in ps.items() if k.startswith("wit 0 ") and "wit.hit" in Z.classes(S, [unit], {x})[x.hex()])
    got = Z.scanned(S, [unit], {"verifier_found": [hit, "adv salt 0 @0"], "leaked": [hit]}, blobs)
    assert "wit.hit" in got["found_classes"][hit] and got["leaked"] == [hit]
    assert got["found_classes"]["adv salt 0 @0"] == []


def test_the_carry_rows_are_each_one_piece_under_zk_scans_rule(tmp_path):
    """Each unit's carry rows (12 bytes) are one piece each, the public priors' ``cin`` not among them. A restart's zero
    ``cin`` row found where the statement's bytes hold zeros is excluded, with no byte count, and a carry row found off
    them is leaked; the control holds every row."""
    zero, live, sent = [0] * 6, [58, 50, 58, 0, 392, 48], [1, 2, 3, 4, 5, 6]
    units = [SimpleNamespace(words={"cin": zero, "cout": live}, chained=False),
             SimpleNamespace(words={"cin": live, "cout": sent}, chained=True)]
    assert list(Z.carry_rows(units, SimpleNamespace(public=True))) == ["cout 0", "cin 0", "cout 1"]
    carry = Z.carry_rows(units, SimpleNamespace(public=False))
    assert carry == {"cout 0": Z._u16(live), "cin 0": bytes(12), "cout 1": Z._u16(sent), "cin 1": Z._u16(live)}
    vdir = tmp_path / "verifier"
    vdir.mkdir()
    (vdir / "session-s-circuit.rep0.bin").write_bytes(bytes(64) + Z._u16(sent))
    (vdir / "pub-2.bin").write_bytes(b"\x07" + bytes(20))
    (tmp_path / "inst.bin").write_bytes(b"".join(carry.values()))
    got = Z.ZS.received(vdir, tmp_path / "inst.bin", carry, set(carry))
    assert got["verifier_found"] == ["cin 0 @0", "cout 1 @0"]
    assert got["excluded"] == {"cin 0 @0": {"hex": "00" * 12, "public": ["pub"]}}
    assert got["leaked"] == ["cout 1 @0"] and got["control"]["holds_all"]

    (vdir / "pub-2.bin").write_bytes(bytes(range(1, 22)))
    assert Z.ZS.received(vdir, tmp_path / "inst.bin", carry, set(carry))["leaked"] == ["cin 0 @0", "cout 1 @0"]
