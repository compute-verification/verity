"""a5's header check of a member reads the registered file the Lean executable is given for it (`a5.header_why`, at
`Member.reads`, `Member.held`): a port the slice holds no root of, read at a registered value the file holds there, is
held by that read, and read at any other value refused by name (`members.unregistered_why`); a wired row the slice
holds at its writer's field tree is held only by the file's read of it there (`members.shared_unread`)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.protocols.proof.spot.verifier.one_stage import members as MB

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "one_stage"))
import a5 as A5

SLICE = {"population": 2, "scheme": "s", "roots": [{"domain": "dx", "port": "x", "root": "rx"}]}
WIRED = {"population": 2, "scheme": "s", "roots": [{"domain": "dw", "port": "x", "root": "rw", "tree": "W/o0"}]}


def _header(ports: dict[str, str]) -> dict:
    return {"frame_v3": {"domain_ids": {p: f"d{p}" for p in ports}, "roots": dict(ports)}, "instances": 2,
            "scheme": "s", "units": {"indices": [0, 1]}}


def _file(**ports: str) -> dict:
    return {"format": "verity/registered-reads/v2",
            "ports": {p: {"domain": "d" * 128, "leaves": 2, "root": "a" * 128, "value": v} for p, v in ports.items()}}


def _why(slice_: dict, header: dict, own: dict | None) -> str | None:
    m = SimpleNamespace(key="t/c0", base=0, n=2, outs=[], port=lambda p: p)
    mine = SimpleNamespace(law=None, tree=lambda m, p: p)
    return A5.header_why(mine, {}, ([("t", [[0, 2]])], {"t/c0": slice_}), {"m": m, "header": header}, own)


def test_a_registered_read_holds_its_port_and_any_other_read_is_refused_by_name():
    header = _header({"x": "rx", "w": "rw"})
    assert _why(SLICE, header, _file(w="h/w")) is None
    for v in ("committed/w", "input/n3", "t/o0"):
        assert _why(SLICE, header, _file(w=v)) == MB.unregistered_why("w", v)
    assert _why(SLICE, header, None) == MB.unwired_why("w")


def test_a_wired_row_is_held_only_by_the_files_read_of_it_at_its_writers_tree():
    header = _header({"x": "rx"})
    assert _why(WIRED, header, _file()) == MB.shared_unread("x", "W/o0")
    assert _why(WIRED, header, None) == MB.shared_unread("x", "W/o0")
    assert _why(WIRED, header, _file(x="W/o0")) is None


def test_an_audit_naming_tied_parameters_says_first_that_no_test_held_their_shared_root(tmp_path, capsys):
    out = tmp_path / "out"
    out.write_text("")
    with pytest.raises(FileExistsError):
        A5.audit(SimpleNamespace(tie={"tied": ["s", "a"]}, window_root=None), {}, out, None, out, dict, 0, {})
    assert capsys.readouterr().out.splitlines() == [
        "TIED-ROOT [\"a\", \"s\"]: none held it: no test holds a tied node's shared root to the record"]
    with pytest.raises(FileExistsError):
        A5.audit(SimpleNamespace(tie={}, window_root=None), {}, out, None, out, dict, 0, {})
    assert capsys.readouterr().out == ""
