"""compile_slice_zk proves w212 at its run's engine class, scheduler included, and its scan of what the verifier
received judges every found witness piece and carry row by the lane's one rule (`leak_scan.scan`), with no byte-count
exemption. Nothing here lowers or proves (`compile_slice_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
#: `a8.find`'s answer, by substring: each piece (named by its needles) found in `data`
A8 = SimpleNamespace(find=lambda data, table: {v: 1 for k, v in table.items() if k in data})


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_compile_slice_zk", HERE / "compile_slice_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_w212_is_proved_at_its_runs_engine_class(monkeypatch):
    """8eb1's engine class is its trace header's, AsyncScheduler (AS = 1) included, and w212 is loaded at it. A fixture
    at another scheduler, the sync class's (AS = 0), is refused, so every window of the run is proved at its one AS."""
    Z = _load()
    x = Z.load("w212")
    assert Z.engine_class()["async_scheduling"] is True
    assert x.fx["engine"] == Z.engine_class() and x.st["AS"] == 1 and x.sl.steps == 21
    monkeypatch.setattr(Z, "W212", Z.FIXTURES / "compile_slice_w212_class.json")
    with pytest.raises(SystemExit, match="isn't the run's engine class"):
        Z.load("w212")


def test_a_piece_with_one_nonzero_byte_is_judged_by_the_rule(tmp_path):
    """No piece is left out for its byte count: a witness piece with one nonzero byte found in a file whose format isn't
    read is leaked, and one whose bytes the public rows hold is structured."""
    Z = _load()
    low, pub = b"\x05" + bytes(15), bytes(15) + b"\x80"
    vdir = tmp_path / "verifier"
    vdir.mkdir()
    (vdir / "openings.json").write_bytes(b"[" + low + b"]")
    (vdir / "pub-2.bin").write_bytes(b"\x01" + pub)
    (tmp_path / "inst.bin").write_bytes(low + pub)
    got = Z.received(A8, vdir, tmp_path / "inst.bin", {"wit 0 @0": low, "wit 8 @0": pub}, {"wit 0 @0", "wit 8 @0"})
    assert got["verifier_found"] == ["wit 0 @0", "wit 8 @0"]
    assert got["leaked"] == ["wit 0 @0"] and got["structured"] == {"wit 8 @0": "public"}
    assert got["control"]["holds_all"]


def test_the_carry_rows_are_each_scanned_whole_under_the_rule(tmp_path):
    """Each unit's carry rows (12 bytes) are scanned whole, the public priors' ``cin`` not among them. A restart's zero
    ``cin`` row is structured where the public rows hold its bytes, and leaked where they don't; a carry row found off
    them is leaked; the control holds every row."""
    Z = _load()
    zero, live, sent = [0] * 6, [58, 50, 58, 0, 392, 48], [1, 2, 3, 4, 5, 6]
    units = [SimpleNamespace(words={"cin": zero, "cout": live}, chained=False),
             SimpleNamespace(words={"cin": live, "cout": sent}, chained=True)]
    assert list(Z.carry_rows(units, SimpleNamespace(public=True))) == ["cout 0", "cin 0", "cout 1"]
    carry = Z.carry_rows(units, SimpleNamespace(public=False))
    assert carry == {"cout 0": Z._u16(live), "cin 0": bytes(12), "cout 1": Z._u16(sent), "cin 1": Z._u16(live)}
    vdir = tmp_path / "verifier"
    vdir.mkdir()
    (vdir / "openings.json").write_bytes(bytes(64) + Z._u16(sent))
    (vdir / "pub-2.bin").write_bytes(b"\x07" + bytes(20))
    (tmp_path / "inst.bin").write_bytes(b"".join(carry.values()))
    got = Z.received(A8, vdir, tmp_path / "inst.bin", carry, set(carry))
    assert got["verifier_found"] == ["cin 0", "cout 1"]
    assert got["structured"] == {"cin 0": "public"} and got["leaked"] == ["cout 1"] and got["control"]["holds_all"]

    (vdir / "pub-2.bin").write_bytes(bytes(range(1, 22)))
    assert Z.received(A8, vdir, tmp_path / "inst.bin", carry, set(carry))["leaked"] == ["cin 0", "cout 1"]
