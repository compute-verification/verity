"""Sampled proofs' answer on a served window from what the records audit wrote (`served_integrity.answer`):

* an accepted audit record is sampled proofs' accept: the registration's digest, the drawn and proved units with the bound
  on wrong units at δ, and the profile as `Core.Integrity` with δ an exact ratio, all read back by the core answer's
  decoder; the report's ε is that bound over the window's units;
* a refusal of the records, a check before the sample that failed, an audit record that doesn't accept and advice fixed
  after the sample each reject by name, with no public outputs or profile;
* an audit that left nothing behind is an error, not a verdict;
* an audit record decided at δ = 0.01 (a float in the record) is the exact ratio 1/100 in the profile.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.sampled_proofs.one_stage import registration as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import served_integrity as I  # noqa: E402
from served_verdicts import check_answer, epsilon  # noqa: E402

REG = {"format": R.FORMAT, "protocol": R.PROTOCOL, "program": "aa" * 64, "partition": "bb" * 64, "population": 1024,
       "law": {"law": "subset", "k": 2}}
DRAW = {"law": "subset", "k": 2, "population": 1024, "units": [700, 5]}


def write(out: Path, summary: dict, record: dict | None = None) -> Path:
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(summary))
    if record is not None:
        (out / "window").mkdir()
        (out / "window" / "audit.json").write_text(json.dumps(record))
    return out


def record(accepted: bool, **checks) -> dict:
    return {"accepted": accepted, "registration": REG, "draw": DRAW, "proved_units": [700, 5],
            "checks": {"draw": {"ok": True}, **{k: {"ok": v} for k, v in checks.items()}},
            "wrong_units": {"delta": 0.5, "of": 1024, "bound": 3} if accepted else None}


def test_an_accepted_audit_is_an_accept(tmp_path):
    out = write(tmp_path / "a", {"checks_before_draw": {"registration": True}, "fixed_before_draw": {"ok": True}},
                record(True))
    a = I.answer(out, REG)
    assert check_answer(a) == a
    assert a["verdict"] == "accept" and a["protocol"] == "verity/sampled-proofs/one-stage/v1"
    assert a["commitment"] == R.digest(REG)
    assert a["public"] == {"bound": 3, "drawn": [5, 700], "per_template": None, "proved": [5, 700]}
    p = CC.integrity(a["profile"])
    assert str(p["delta"]) == "1/2" and p["drawn"] == [("unit", 2)]
    assert p["levels"] == [{"name": "unit", "query": "bb" * 64, "size": 1024, "draw": {"subset": 2}}]
    assert epsilon(a) == "3/1024"


@pytest.mark.parametrize("summary,rec,why", [
    ({"refused": {"records": "ValueError: the window has 2 row tables"}}, None, "refused: "),
    ({"checks_before_draw": {"registration": False, "m0": True}}, None, "before the draw: registration"),
    ({"checks_before_draw": {"registration": True}}, record(False, proved=False), "audit record: proved"),
    ({"checks_before_draw": {"registration": True}, "audit": {"accepted": False, "outcome": "coins refused"}}, None,
     "no audit record: "),
    ({"fixed_before_draw": {"ok": False, "codes": ["after-the-draw"]}}, record(True), "not fixed before the draw: after"),
])
def test_refusals_reject_by_name(tmp_path, summary, rec, why):
    a = I.answer(write(tmp_path / "a", summary, rec), REG)
    assert check_answer(a) == a
    assert a["verdict"] == "reject" and a["reason"].startswith(why)
    assert a["commitment"] == R.digest(REG)


def test_a_refused_audit_record_names_leans_refusals(tmp_path):
    why = {f"lean-verify/m{k}": f"registered reads: port o0: table row 0 does not open out's registered root at "
                                f"position {k}" for k in range(5)}
    summary = {"checks_before_draw": {"registration": True}, "audit": {"why": {**why, "draw": None}}}
    a = I.answer(write(tmp_path / "a", summary, record(False, proved=False)), REG)
    assert a["verdict"] == "reject" and a["reason"].startswith("audit record: proved; lean-verify/m0: registered reads")
    assert "position 2" in a["reason"] and "position 3" not in a["reason"] and a["reason"].endswith("and 2 more")


def test_a_tampered_wired_window_names_the_drawn_units_its_sessions_refused_before_leans_missing_files(tmp_path):
    """A drawn member whose live session the verifier of record rejects leaves no proof files, so Lean's replay finds
    none: the reason names the sampled proof blocks not proved, by template, and the verifier of record's rejection first."""
    mm, quant = "ScaledMmS8_v1{K=128,N=128}", "QuantRowS8_v1{N=128,RELU=true}"
    rec = record(False, proved=False)
    rec["checks"]["proved"]["detail"] = {"units": {"drawn": list(range(12)), "statement": list(range(8, 12))}}
    rec["draw"] = {**DRAW, "strata": [{"template": mm, "units": [[0, 8]]}, {"template": quant, "units": [[8, 12]]}]}
    why = {**{f"lean-verify/m{k}": "session files: no such file or directory" for k in range(4)},
           **{f"m0-verifier/m{k}": "R7: no proof for circuit/rep0 (lone rep)" for k in range(4)}}
    a = I.answer(write(tmp_path / "a", {"checks_before_draw": {"registration": True}, "audit": {"why": why}}, rec), REG)
    assert a["verdict"] == "reject"
    assert a["reason"].startswith(f"audit record: proved (drawn units not proved: {mm} 0-7); m0-verifier/m0: R7: ")
    assert "session files" not in a["reason"] and quant not in a["reason"] and a["reason"].endswith("and 5 more")


def test_an_audit_that_left_nothing_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="no summary"):
        I.answer(tmp_path, REG)
    with pytest.raises(RuntimeError, match="no audit record and no refusal"):
        I.answer(write(tmp_path / "a", {"checks_before_draw": {"registration": True}}), REG)


def test_an_audit_at_one_percent_is_the_exact_ratio(tmp_path):
    rec = {**record(True), "wrong_units": {"delta": 0.01, "of": 1024, "bound": 69}}
    a = I.answer(write(tmp_path / "a", {"checks_before_draw": {"registration": True}}, rec), REG)
    assert a["profile"]["delta"] == "1/100" and str(CC.integrity(a["profile"])["delta"]) == "1/100"
