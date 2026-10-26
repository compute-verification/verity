"""The coverage table (data/cover/, MODEL-COVER): every opcode of NVIDIA's family tables, every SASS form of the corpus and
every census kind on the H200 and the RTX PRO 6000 has a row, every row names a real envelope op and θ key, and every
dominated tensor row's rate times its weight is at most the card's tensor rate."""
from __future__ import annotations

import re
import shutil
import tomllib

import pytest

from verity_machine_model import cover as C
from verity_machine_model.cli import main


@pytest.fixture(scope="module")
def cover() -> C.Cover:
    return C.load()


def test_every_opcode_form_and_kind_has_a_row(cover):
    assert C.problems(cover) == []


def test_the_census_and_both_tables_are_read(cover):
    for dev in C.DEVICES:
        srcs = set().union(*cover.opcodes[dev].values())
        assert srcs == {"BU 12.9", "CUDA 13", "corpus"}, dev
        assert len(cover.forms[dev]) > 900 and len(cover.kinds[dev]) >= 50
    assert {"QMMASF16832", "OMMASF16864", "HMMA16816F16"} <= cover.kinds["sm_120"]
    assert {"BMMA168256_B1", "DMMA16816"} <= cover.kinds["sm_90"]


def _copy(tmp_path):
    d = tmp_path / "cover"
    shutil.copytree(C.COVER, d)
    return d


def _drop_op(d, op: str) -> None:
    p = d / "rows.toml"
    p.write_text(re.sub(rf'"{op}",\s*', "", p.read_text()))


def test_an_opcode_without_a_row_fails(tmp_path):
    d = _copy(tmp_path)
    _drop_op(d, "FCHK")
    ps = C.problems(C.load(d))
    assert any(p.startswith("sm_90 FCHK ") and "no plain row" in p for p in ps)
    assert any(p.startswith("sm_120 FCHK ") for p in ps)


def test_a_tensor_row_over_the_tensor_rate_fails(tmp_path):
    d = _copy(tmp_path)
    p = d / "envelope.toml"
    p.write_text(p.read_text().replace('"16" = 2, tf32 = 4', '"16" = 3, tf32 = 4', 1))
    ps = C.problems(C.load(d))
    assert any("HMMA" in p and "over ρ_T = 1024" in p for p in ps)


def test_a_form_two_rows_claim_fails(tmp_path):
    d = _copy(tmp_path)
    p = d / "rows.toml"
    p.write_text(p.read_text().replace("forms = ['IMMA\\.16816\\..*', 'IMMA\\.8816\\..*']",
                                       "forms = ['IMMA\\.16816\\..*', 'IMMA\\.8816\\..*', 'IMMA\\.16832\\..*']"))
    assert any("IMMA.16832.S8.S8 matches the forms of rows" in p for p in C.problems(C.load(d)))


def test_every_hole_says_its_severity_and_the_known_holes_are_closed(cover):
    for r in cover.rows:
        assert r.mark != "hole" or r.raw.get("severity"), r.ops
    closed = {"HMMA.16816.F16": "covered", "QMMA.SF.16832.F32.E4M3.E2M1.E8": "covered",
              "QMMA.SP.16864.F32.E4M3.E4M3": "covered"}
    for form, mark in closed.items():
        assert C.row_of(cover, "sm_120", form).mark == mark, form
    assert C.row_of(cover, "sm_120", "OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X").envelope == "T.sp"
    assert C.row_of(cover, "sm_90", "BMMA.168256.XOR.POPC").mark == "covered"


def test_envelope_ops_are_the_brief_list_made_precise():
    env = tomllib.loads((C.COVER / "envelope.toml").read_text())
    assert [o["key"] for o in env["op"]] == ["T", "T.sp", "L", "M", "M.bcast", "M.red", "S", "H", "-"]
    for dev in C.DEVICES:
        assert env["device"][dev]["rho"]["lane"] == 128


def test_the_command_checks_and_renders(capsys):
    assert main(["cover", "--check"]) == 0
    assert "has a row" in capsys.readouterr().out
    assert main(["cover", "--card", "sm_120"]) == 0
    out = capsys.readouterr().out
    assert "## sm_120" in out and "| QMMA | `QMMA\\.SF\\..*`" in out and "Holes (each a gap" in out
    assert "## sm_90" not in out
