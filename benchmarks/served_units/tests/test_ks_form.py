"""ks_form.py on statements staged under FLOCK_WORK/data with synthetic circuit heads (the circuit file's first lines are all
it reads): one line a finished staging (``.done``), its type from stage.json (a census name, else stage.py's type), and the
mirror's verdict: accepted, a shared-row public file's open check, the scope refusal (a tail net returning the outputs, a
unit of two input groups), and the template's (no unit a block, outputs past the widest size class)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import load


def head(meta: dict, groups: int = 1, before: int = 0) -> str:
    tail = "CIRCUIT stage0 2\nAND 0 1\nAND 1 2\n" * before
    return f"flock-circuit/v2\nMETA {json.dumps(meta)}\n{tail}CIRCUIT unit 3\nflock-ir-unit/v2 u 8 0 {groups}\nAND 0 1\n"


def staged(fw: Path, name: str, meta: dict, *, groups: int = 1, before: int = 0, shared: bool = False,
           done: bool = True, rec: dict | None = None) -> None:
    d = fw / "data" / name
    d.mkdir(parents=True)
    (d / "circuit.txt").write_text(head(meta, groups, before))
    (d / "pub.bin").write_bytes(json.dumps({"format": "p", **({"shared_rows": [0]} if shared else {})}).encode()
                                + b"\n\x00\x01")
    (d / "stage.json").write_text(json.dumps(rec or {"census": f"T_v1{{{name}}}", "n": 8}))
    if done:
        (d / ".done").write_text("")


@pytest.fixture
def ks(monkeypatch):
    return load(monkeypatch, "ks_form.py")


def test_lines(ks, tmp_path, capsys):
    ok = {"out_net": "unit", "units_per_vu": 2, "leaves_out": [[0] * 8, [0] * 8]}
    fw, also = tmp_path / "fw", tmp_path / "also"
    staged(fw, "a-ok", ok)
    staged(fw, "b-tail", {**ok, "out_net": "stage0"}, before=1)
    staged(fw, "c-groups", ok, groups=2)
    staged(fw, "d-shared", ok, shared=True)
    staged(fw, "e-norow", {**ok, "units_per_vu": 0})
    staged(fw, "f-wide", {"units_per_vu": 1, "leaves_out": [[0] * (ks.PF.UNIT_CLASS_MAX // 16 + 1)]})
    staged(fw, "g-staging", ok, done=False)
    staged(also, "h-part", ok, before=2, rec={"unit": "gq", "n": 4096})
    ks.main([str(fw), str(also)])
    lines = [ln.split("\t") for ln in capsys.readouterr().out.splitlines()]
    assert lines[0] == ["dir", "type", "n", "out_net", "shared_rows", "mirror_zk"]
    got = {r[0]: r[1:] for r in lines[1:]}
    scope = ks.PF._refusal(ks.PF.SCOPE_WHY)
    assert got == {
        "a-ok": ["T_v1{a-ok}", "8", "unit", "false", "accepted"],
        "b-tail": ["T_v1{b-tail}", "8", "stage0", "false", scope],
        "c-groups": ["T_v1{c-groups}", "8", "unit", "false", scope],
        "d-shared": ["T_v1{d-shared}", "8", "unit", "true", ks.OPEN],
        "e-norow": ["T_v1{e-norow}", "8", "unit", "false", ks.PF.NO_ROW_WHY],
        "f-wide": ["T_v1{f-wide}", "8", "unit", "false", ks.PF.WIDE_WHY],
        "h-part": ["gq", "4096", "unit", "false", "accepted"],
    }
    assert [r[0] for r in lines[1:]] == [*sorted(got)], "one line a statement, FLOCK_WORK by FLOCK_WORK, by name"


def test_no_meta(ks, tmp_path):
    """A circuit file whose second line is no META has no out_net to read, and the mirror leaves it to Lean."""
    d = tmp_path / "data" / "x"
    staged(tmp_path, "x", {})
    (d / "circuit.txt").write_text("flock-circuit/v1\nCIRCUIT unit 1\nAND 0 1\n")
    assert ks.form(d) == ["x", "T_v1{x}", "8", "-", "false", "accepted"]
