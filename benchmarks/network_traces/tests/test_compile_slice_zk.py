"""compile_slice_zk proves w212 at its run's engine class, scheduler included, and its scan of what the verifier
received judges every found witness piece and carry row by the lane's one rule (`leak_scan.scan`), with no byte-count
exemption, each row shorter than the scan's 16-byte piece found whole. What `prove` reads off its helpers exists, and
`cost3` takes CompileSlice_v3's classes and fixtures. Nothing here lowers or proves (`compile_slice_zk.prove` and
`cost3` do, on a node)."""

from __future__ import annotations

import importlib.util
import json
import re
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


def test_the_rows_shorter_than_a_piece_are_found_whole_under_the_rule(tmp_path):
    """`scan_pieces` cuts each row of at least 16 bytes whole (`leak_scan.public_pieces`) and keeps each shorter row
    whole: the ``srv`` rows of a few steps (10 bytes at the demo's G = 2, 6 at G = 1), which `leak_scan.cut` refuses
    and the aligned cutter before it never scanned. Such a row is found raw or as hex, a raw one under 8 bytes too
    (`a8.find` keys each needle on its first 8 bytes, so `_find` searches those itself), and judged by the rule: leaked
    off the statement's bytes, structured once a ``pub`` row holds them."""
    Z = _load()
    a8 = SimpleNamespace(PIECE=16,
                         needles=lambda pieces: {nd: k for k, x in pieces.items() for nd in (x, x.hex().encode())},
                         find=lambda data, table: {k: 1 for nd, k in table.items() if len(nd) >= 8 and nd in data})
    wit, srv0, srv1 = list(range(1, 19)), [7, 9, 11, 13, 15], [21, 23, 25]
    units = [SimpleNamespace(words={"wit": wit, "srv0": srv0, "srv1": srv1}, chained=False)]
    with pytest.raises(ValueError, match="shorter than the scan's 16-byte piece"):
        Z.LS.public_pieces(a8, {"srv0 0": Z._u16(srv0)}, set(), b"")
    pieces, held, public = Z.scan_pieces(a8, units, {}, SimpleNamespace(public=False), ("srv0", "srv1"), b"")
    assert pieces == {"wit 0 @0": Z._u16(wit)[:16], "wit 0 @16": Z._u16(wit)[16:32], "wit 0 @20": Z._u16(wit)[20:],
                      "wit 0 @32": Z._u16(wit)[32:].ljust(16, b"\0"), "srv0 0": Z._u16(srv0), "srv1 0": Z._u16(srv1)}
    assert held == set(pieces) and public == []
    vdir = tmp_path / "verifier"
    vdir.mkdir()
    (vdir / "openings.json").write_bytes(Z._u16(srv0).hex().encode() + b"," + Z._u16(srv1) + b"\x01\x02")
    (tmp_path / "inst.bin").write_bytes(Z._u16(wit).ljust(48, b"\0") + Z._u16(srv0) + Z._u16(srv1))
    got = Z.received(a8, vdir, tmp_path / "inst.bin", pieces, held)
    assert got["verifier_found"] == ["srv0 0", "srv1 0"] and got["leaked"] == ["srv0 0", "srv1 0"]
    assert got["control"]["holds_all"] and got["control"]["pieces_found"] == 6
    (vdir / "pub-2.bin").write_bytes(Z._u16(srv1) + bytes(4))
    got = Z.received(a8, vdir, tmp_path / "inst.bin", pieces, held)
    assert got["leaked"] == ["srv0 0"] and got["structured"] == {"srv1 0": "public"}


def test_what_prove_reads_off_its_helpers_is_there():
    """`prove`, `rescan` and `cost3` run on a node only, and read the one-stage namespace (``M``, `warden_zk._one_stage`),
    its modules and `warden_zk` by attribute: each attribute the module reads off them exists, so a helper renamed under it
    fails here rather than on the node."""
    Z = _load()
    src = (HERE / "compile_slice_zk.py").read_text()
    M = Z.WZ._one_stage()
    missing = [f"WZ.{k}" for k in set(re.findall(r"\bWZ\.(\w+)", src)) if not hasattr(Z.WZ, k)]
    missing += [f"M.{k}" for k in set(re.findall(r"\bM\.(\w+)", src)) if not hasattr(M, k)]
    missing += [f"M.{k}.{j}" for k, j in set(re.findall(r"\bM\.(\w+)\.(\w+)", src))
                if hasattr(M, k) and not hasattr(getattr(M, k), j)]
    assert not missing


def test_cost3_takes_v3s_demo_class_dry(tmp_path):
    """`cost3 --dry` on CompileSlice_v3's demonstration class: the reference's rows registered, the unit accepts."""
    Z = _load()
    assert Z.main(["cost3", "--class", "demo", "--dry", "--out", str(tmp_path)]) == 0
    got = json.loads((tmp_path / "summary.json").read_text())
    assert got["verdict"] == Z.V3.ACCEPT and got["unit"] == "CompileSlice_v3" and got["arms"] == ["fw"]
    with pytest.raises(SystemExit, match="isn't fw, nofw or plain"):
        Z.main(["cost3", "--class", "demo", "--dry", "--arms", "gpu", "--out", str(tmp_path)])


def test_a_fixture_is_a_v3_class_with_its_pool_statics(tmp_path):
    """A ``--fixture`` in the corpus fixture's layout, holding CD and SW, loads as its class and slice (here the demo's);
    one without its pool statics is refused, naming them."""
    from verity.ml.protocols.certifier import compile_slice_v2 as V2

    Z = _load()
    x = Z.load3("demo")
    fx = {"engine": V2.DEMO_ENGINE, **{k: x.st[k] for k in ("R", "W", "A", "IW", "Q", "D", "P", "K", "CN", "CTW", "CU",
                                                            "CD", "SW")},
          "arrivals": x.sl.arrivals, "decisions": x.sl.decisions, "steps": x.sl.steps, "tokens": x.sl.tokens}
    f = tmp_path / "fixture.json"
    f.write_text(json.dumps(fx))
    y = Z.load3("corpus", f)
    assert (y.st, y.sl) == (x.st, x.sl) and y.name == str(f)
    f.write_text(json.dumps({k: v for k, v in fx.items() if k not in ("CD", "SW")}))
    with pytest.raises(SystemExit, match="holds no CD, SW"):
        Z.load3("corpus", f)
