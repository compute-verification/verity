"""compile_slice_zk proves w212 at its run's engine class, scheduler included, and its scan of what the verifier
received holds the rows it finds whole, those shorter than a piece, to the rule every zero-knowledge driver's scan follows
(`leak_scan`), with no byte-count exemption. Nothing here lowers or proves (`compile_slice_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
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


def test_the_rows_shorter_than_a_piece_are_scanned_whole_under_the_leak_rule(tmp_path):
    """The rows `leak_scan.cut` can't cut, the carry rows (12 bytes) and the ``srv`` rows of a few steps (10 bytes at
    the demo's G = 2), are found whole and held to `leak_scan`'s rule like any piece, with no byte-count exemption. A
    restart from zero counts commits a ``cin`` row of twelve zero bytes: found in a file whose format isn't read it is
    leaked, and it is excused only once the statement (a ``pub`` row) holds its bytes, while a row the statement doesn't
    hold stays leaked. The control is held to every row."""
    Z = _load()
    a8 = SimpleNamespace(find=lambda data, table: {v: 1 for k, v in table.items() if k in data}, PIECE=16)
    zero, live, sent = [0] * 6, [58, 50, 58, 0, 392, 48], [1, 2, 3, 4, 5, 6]
    units = [SimpleNamespace(words={"cin": zero, "cout": live}, chained=False),
             SimpleNamespace(words={"cin": live, "cout": sent}, chained=True)]
    carry = Z.carry_rows(units, SimpleNamespace(public=False))
    assert list(carry) == ["cout 0", "cin 0", "cout 1", "cin 1"] and {len(x) for x in carry.values()} == {12}
    assert list(Z.carry_rows(units, SimpleNamespace(public=True))) == ["cout 0", "cin 0", "cout 1"]
    srv = Z._u16([7, 9, 11, 13, 15])
    with pytest.raises(ValueError, match="shorter than the scan's 16-byte piece"):
        Z.LS.public_pieces(a8, {"srv1 0": srv}, set(), b"")
    vdir = tmp_path / "verifier"
    vdir.mkdir()
    (vdir / "proof.bin").write_bytes(bytes(64) + Z._u16(sent) + srv.hex().encode())
    (tmp_path / "control.bin").write_bytes(Z._u16(zero) + Z._u16(live) + Z._u16(sent) + srv)

    def scan():
        return Z.found(a8, vdir, tmp_path / "control.bin", {**carry, "srv1 0": srv})

    got = scan()
    assert (got["distinct_rows"], got["verifier_found"], got["control_holds_all"]) == (
        4, ["cin 0", "cout 1", "srv1 0"], True)
    assert got["leaked"] == ["cin 0", "cout 1", "srv1 0"] and got["structured"] == {}
    assert got["placed"]["cin 0"][0] == ("proof.bin", 0, "file")
    (vdir / "pub-2.bin").write_bytes(bytes(32))
    got = scan()
    assert got["leaked"] == ["cout 1", "srv1 0"] and got["structured"] == {"cin 0": "public"}
