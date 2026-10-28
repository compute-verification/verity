"""The coverage table (data/cover/, MODEL-COVER): every opcode of NVIDIA's family tables, every SASS form of the corpus and
every census kind on the H200 and the RTX PRO 6000 has a row; every row names a real envelope op (MODEL-MIN's `Accel.Op`
or a proposed one), pipe, format and θ key; every lane and collective reads and writes no more words than its op allows;
and every dominated row's rate, in its op's units, is at most its resource's ρ."""
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


def _edit(d, name: str, old: str, new: str) -> None:
    p = d / name
    text = p.read_text()
    assert old in text, old
    p.write_text(text.replace(old, new, 1))


def test_a_missing_fetched_table_names_the_fetch(tmp_path):
    d = _copy(tmp_path)
    (d / "corpus.json").unlink()
    with pytest.raises(FileNotFoundError, match="uv run research data fetch-fixtures benchmarks/pouw/machine_model"):
        C.load(d)


def test_an_opcode_without_a_row_fails(tmp_path):
    d = _copy(tmp_path)
    p = d / "rows.toml"
    p.write_text(re.sub(r'"FCHK",\s*', "", p.read_text()))
    ps = C.problems(C.load(d))
    assert any(p.startswith("sm_90 FCHK ") and "no plain row" in p for p in ps)
    assert any(p.startswith("sm_120 FCHK ") for p in ps)


def test_a_tensor_row_over_the_tensor_rate_fails(tmp_path):
    d = _copy(tmp_path)
    _edit(d, "envelope.toml", "f16 = 2, bf16 = 2, tf32 = 4", "f16 = 3, bf16 = 2, tf32 = 4")
    ps = C.problems(C.load(d))
    assert any("HMMA" in p and "on sm_120" in p and "over ρ tensor = 1024" in p for p in ps)


def test_a_pipe_rate_over_its_rho_fails(tmp_path):
    d = _copy(tmp_path)
    _edit(d, "envelope.toml", "alu = 64, imad = 64", "alu = 32, imad = 64")
    ps = C.problems(C.load(d))
    assert any("LOP3" in p and "on sm_120 θ alu.lop3 is 64" in p and "over ρ alu = 32" in p for p in ps)


def test_the_l2_atomics_at_the_l1_ports_sheet_fail_on_sm_120(tmp_path):
    d = _copy(tmp_path)
    _edit(d, "rows.toml", 'rate_value = 2.33\nrate_unit = "words/SM/clk"', 'rate = "l2.red"')
    ps = C.problems(C.load(d))
    assert any("ATOM" in p and "on sm_120 θ l2.red is 32" in p and "over ρ l2 = 5" in p for p in ps)


def test_a_lane_reading_more_words_than_its_pipe_fails(tmp_path):
    d = _copy(tmp_path)
    _edit(d, "rows.toml", 'pipe = "fma"\nwords = [3, 2]\nrate = ["ffma"', 'pipe = "fma"\nwords = [5, 2]\nrate = ["ffma"')
    ps = C.problems(C.load(d))
    assert any("FFMA" in p and "5 words in and 2 out a lane, past lane's 4 in and 2 out on fma" in p for p in ps)


def test_a_form_two_rows_claim_fails(tmp_path):
    d = _copy(tmp_path)
    _edit(d, "rows.toml", "forms = ['IMMA\\.16816\\..*', 'IMMA\\.8816\\..*']",
          "forms = ['IMMA\\.16816\\..*', 'IMMA\\.8816\\..*', 'IMMA\\.16832\\..*']")
    assert any("IMMA.16832.S8.S8 matches the forms of rows" in p for p in C.problems(C.load(d)))


def test_every_hole_says_its_severity_and_the_known_holes_are_closed(cover):
    for r in cover.rows:
        assert r.mark != "hole" or r.raw.get("severity"), r.ops
    closed = {"HMMA.16816.F16": "covered", "QMMA.SF.16832.F32.E4M3.E2M1.E8": "covered",
              "QMMA.SP.16864.F32.E4M3.E4M3": "covered", "MATCH.ANY.U64": "covered", "LDGMC.E.ADD.F32": "covered"}
    for form, mark in closed.items():
        assert C.row_of(cover, "sm_120", form).mark == mark, form
    assert C.row_of(cover, "sm_120", "OMMA.SF.SP.168128.F32.E2M1.E2M1.UE4M3.4X").envelope == "mmaSp"
    assert C.row_of(cover, "sm_90", "BMMA.168256.XOR.POPC").mark == "covered"
    for form in ("MATCH.ANY", "MATCH.ANY.U64", "REDUX.SUM", "VOTE.ANY", "VOTEU.ALL"):
        assert C.row_of(cover, "sm_90", form).mark == "covered", form


def test_rows_on_proposed_entries_are_holes_on_accel_op_alone(cover):
    fp8 = C.row_of(cover, "sm_120", "QMMA.16832.F32.E4M3.E4M3")
    fp6 = C.row_of(cover, "sm_120", "QMMA.16832.F32.E3M2.E2M3")
    assert C.rests_on(cover, fp8) == [] and C.rests_on(cover, fp6) == ["fmt e3m2", "fmt e2m3"]
    assert C.rests_on(cover, C.row_of(cover, "sm_120", "REDUX.SUM")) == ["op collective", "pipe collective"]
    assert C.mark_of(cover, "sm_120", "REDUX") == "covered"
    assert C.mark_of(cover, "sm_120", "REDUX", proposed=False) == "hole"
    assert C.mark_of(cover, "sm_120", "LOP3", proposed=False) == "covered"
    for dev in C.DEVICES:
        assert C.counts(cover, dev, False).opcodes["hole"] > C.counts(cover, dev, True).opcodes["hole"], dev


def test_the_envelope_is_accel_op_and_the_proposed_entries_say_why(cover):
    env = tomllib.loads((C.COVER / "envelope.toml").read_text())
    assert [o["key"] for o in env["op"]] == ["mma", "mmaSp", "lane", "move", "atom", "red", "branch", "none",
                                             "collective", "tex", "-"]
    assert [p["key"] for p in env["pipe"]] == ["alu", "imad", "fma", "shfl", "f64", "sfu", "collective"]
    for e in (*env["op"], *env["pipe"], *env["fmt"]):
        assert "proposed" not in e or "MODEL-MIN" in e["proposed"] or e["key"] == "collective", e["key"]
    for dev in C.DEVICES:
        assert C.rho(cover, dev, "fma") == 128
        assert C.rho(cover, dev, "dram") == pytest.approx(env["device"][dev]["dram"] /
                                                          (env["device"][dev]["sms"] * env["device"][dev]["clock_mhz"]
                                                           * 1e6))


def test_the_command_checks_and_renders(capsys):
    assert main(["cover", "--check"]) == 0
    assert "has a row" in capsys.readouterr().out
    assert main(["cover", "--card", "sm_120"]) == 0
    out = capsys.readouterr().out
    assert "## sm_120" in out and "| QMMA | `QMMA\\.SF\\..*`" in out and "Holes (each a gap" in out
    assert "sm_120 on Accel.Op alone" in out and "Rows resting on a proposed entry:" in out
    assert out.startswith("The ACCEL ops: ") and "| Mark | ACCEL op |" in out and "the card premise" in out
    assert "envelope ISA" not in out and "coverage premise" not in out
    assert "## sm_90" not in out
