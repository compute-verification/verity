"""compile_slice_zk proves w212 at its run's engine class, scheduler included, and its scan of what the verifier
received holds the carry rows to the rule every zero-knowledge driver's scan follows (`leak_scan`), with no byte-count
exemption. Nothing here lowers or proves (`compile_slice_zk.prove` does, on a node)."""

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


def test_the_carry_rows_are_scanned_under_the_leak_rule(tmp_path):
    """A restart from zero counts commits a ``cin`` row of twelve zero bytes. The carry scan holds it to `leak_scan`'s
    rule like any row, with no byte-count exemption: found in a file whose format isn't read it is leaked, and it is
    excused only once the statement (a ``pub`` row) holds its bytes, while a row the statement doesn't hold stays
    leaked. The control is held to every row."""
    Z = _load()
    a8 = SimpleNamespace(find=lambda data, table: {v: 1 for k, v in table.items() if k in data})
    zero, live, sent = [0] * 6, [58, 50, 58, 0, 392, 48], [1, 2, 3, 4, 5, 6]
    units = [SimpleNamespace(words={"cin": zero, "cout": live}, chained=False),
             SimpleNamespace(words={"cin": live, "cout": sent}, chained=True)]
    vdir = tmp_path / "verifier"
    vdir.mkdir()
    (vdir / "proof.bin").write_bytes(bytes(64) + Z._u16(sent))
    (tmp_path / "control.bin").write_bytes(Z._u16(zero) + Z._u16(live) + Z._u16(sent))
    def scan():
        return Z.carry_rows_received(a8, vdir, tmp_path / "control.bin", units, SimpleNamespace(public=False))

    got = scan()
    assert (got["distinct_rows"], got["verifier_found"], got["control_holds_all"]) == (3, ["cin 0", "cout 1"], True)
    assert got["leaked"] == ["cin 0", "cout 1"] and got["structured"] == {}
    assert got["placed"]["cin 0"][0] == ("proof.bin", 0, "file")
    (vdir / "pub-2.bin").write_bytes(bytes(32))
    got = scan()
    assert got["leaked"] == ["cout 1"] and got["structured"] == {"cin 0": "public"}
