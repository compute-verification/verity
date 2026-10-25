"""compile_slice_zk proves w212 at its run's engine class, scheduler included, and its scan of what the verifier
received leaves out exactly the witness pieces and carry rows with at most one nonzero byte, which a proof's encoding
holds whatever the witness, and keeps the rest held as they were. Nothing here lowers or proves
(`compile_slice_zk.prove` does, on a node)."""

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


def test_sparse_pieces_leaves_out_one_nonzero_byte():
    Z = _load()
    zero, low, high = bytes(16), b"\x05" + bytes(15), bytes(15) + b"\x80"
    two, full = b"\x01\x02" + bytes(14), bytes(range(1, 17))
    pieces = {"wit 0 @0": low, "wit 0 @16": two, "wit 0 @32": full, "wit 8 @0": high, "wit salt 0 @0": zero}
    kept, held, out = Z.sparse_pieces(pieces, {"wit 0 @0", "wit 0 @16", "wit 0 @32", "wit 8 @0"})
    assert kept == {"wit 0 @16": two, "wit 0 @32": full}
    assert held == {"wit 0 @16", "wit 0 @32"}
    assert out == sorted(x.hex() for x in (zero, low, high))


def test_carry_rows_received_leaves_out_a_zero_row(tmp_path):
    """A restart from zero counts commits a ``cin`` row of twelve zero bytes, which any proof holds: the carry scan
    lists it as sparse and scans the rest, finding a row the verifier received and holding the control to them all."""
    Z = _load()
    a8 = SimpleNamespace(find=lambda data, table: {v: 1 for k, v in table.items() if k in data})
    zero, live, sent = [0] * 6, [58, 50, 58, 0, 392, 48], [1, 2, 3, 4, 5, 6]
    units = [SimpleNamespace(words={"cin": zero, "cout": live}, chained=False),
             SimpleNamespace(words={"cin": live, "cout": sent}, chained=True)]
    vdir = tmp_path / "verifier"
    vdir.mkdir()
    (vdir / "proof.bin").write_bytes(bytes(64) + Z._u16(sent))
    (tmp_path / "control.bin").write_bytes(Z._u16(live) + Z._u16(sent))
    got = Z.carry_rows_received(a8, vdir, tmp_path / "control.bin", units, SimpleNamespace(public=False))
    assert got == {"distinct_rows": 2, "verifier_found": ["cout 1"], "control_holds_all": True, "sparse": ["cin 0"]}
