"""stage.py's census types: `parse` of census:NAME (the census name verbatim) and of a CENSUS key's statics, and `main`
staging one through `stage_census` on a stub of `class_statement.UnitStager` (nothing is lowered or proved): the serving
rows' template the name resolves, its rows' words, the size ``full`` and ``full/D`` resolve to at the block's k_log (never
under one block's units), and stage.json's record."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from conftest import load

SILU = "SiluMul_v2{I=256}"


@pytest.fixture(scope="module")
def stage():
    with pytest.MonkeyPatch.context() as mp:
        yield load(mp, "stage.py")


def test_parse(stage):
    name, cen, S = stage.parse(f"census:{SILU}")
    assert (name, S, cen.name(S)) == ("census", {}, SILU)
    assert stage.parse("census:Fam")[1].name({}) == "Fam"
    name, cen, S = stage.parse("gq:K=2048")
    assert (name, S, cen.name(S)) == ("gq", {"K": 2048, "G": 128}, "Fp8GroupQuantUe8m0_v1{K=2048,G=128}")
    _, cen, S = stage.parse("mla-split-part")
    assert cen.name(S) == ('SparseMlaSplitPart_v1{T=64,DR=64,DV=512,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v1"},'
                           'INV={"fn":"Fa3InvSum_v1"},SCALE={"f64":"0x1.0000000000000p-4"}}')
    for bad in ("census:", "census:{A=1}", "census:Fam{A=1"):
        with pytest.raises(SystemExit, match="census:NAME takes a census name"):
            stage.parse(bad)
    with pytest.raises(SystemExit, match="gq has no static 'X'"):
        stage.parse("gq:X=1")


class Stager:
    """`UnitStager`'s face to `stage_census`: its META (k_log 32 and 4 units a block, so a statement holds 32 instances),
    the rows it is handed, and the files a staging writes."""

    seen: dict = {}

    def __init__(self, d, name, in_rows, out_rows, circuit, **stmt):
        self.meta = {"k_log": 32, "vus_per_block": 4}
        self.low = SimpleNamespace(layout=SimpleNamespace(counts={"ands": 5, "rows": 6, "nnz_a": 7, "nnz_b": 8}))
        self.pin, self.class_sha512 = "p" * 128, "c" * 128
        Stager.seen.update(definition=d.id, name=name, stmt=stmt, circuit=circuit)

    def stage(self, rows, out, salt_key):
        Stager.seen["rows"] = {r: v.shape for r, v in rows.items()}
        for f in ("circuit.txt", "inst.bin", "pub.bin"):
            (out / f).write_bytes(b"xy")
        return {"meta": {"k_log": 32, "vus_per_block": 4}, "ands": 5, "unit_nnz": 9}


@pytest.mark.parametrize(("spec", "n", "want"), [(f"census:{SILU}", "full", 32), (f"census:{SILU}", "full/4", 8),
                                                 (f"census:{SILU}", "full/64", 4), ("gq:K=2048", "full/2", 16),
                                                 (f"census:{SILU}", "3", 3)])
def test_main(stage, monkeypatch, tmp_path, capsys, spec, n, want):
    monkeypatch.setattr(stage.CS, "UnitStager", Stager)
    monkeypatch.setattr(sys, "argv", ["stage.py", spec, n, str(tmp_path)])
    stage.main()
    rec = json.loads((tmp_path / "stage.json").read_text())
    assert rec == json.loads(capsys.readouterr().out)
    census = stage.parse(spec)[1].name(stage.parse(spec)[2])
    assert (rec["census"], rec["definition"], rec["n"]) == (census, census, want)
    assert (tmp_path / "pin.txt").read_text() == "p" * 128 and rec["class_sha512"] == "c" * 128
    assert rec["nets"] == [{"net": "unit", "ands": 5, "rows": 6, "nnz_a": 7, "nnz_b": 8}]
    assert (rec["k_log"], rec["vus_per_block"], rec["circuit_bytes"], rec["inst_bytes"]) == (32, 4, 2, 2)
    assert Stager.seen["circuit"] == tmp_path / "circuit.txt" and Stager.seen["stmt"] == rec["statement"]
    assert Stager.seen["rows"] == {r: (want, b // 16) for r, _s, b, _ps in rec["in_rows"]}


@pytest.mark.parametrize(("argv", "why"), [([f"census:{SILU}", "full/0"], "invalid literal"),
                                           (["s8-linear", "full"], "is for a CENSUS type"),
                                           (["census:SiluMul_v2", "2"], "lacks the static")])
def test_main_refuses(stage, monkeypatch, tmp_path, argv, why):
    monkeypatch.setattr(stage.CS, "UnitStager", Stager)
    monkeypatch.setattr(sys, "argv", ["stage.py", *argv, str(tmp_path)])
    with pytest.raises((SystemExit, ValueError), match=why):
        stage.main()
    assert not (tmp_path / "stage.json").exists()
